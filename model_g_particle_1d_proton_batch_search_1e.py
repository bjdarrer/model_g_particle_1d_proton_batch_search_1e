#!/usr/bin/env python3
""""
Date: 11 March 2026 16:39 GMT

Proton local-refinement search with neutron-anchor support

Code: "model_g_particle_1d_proton_batch_search_1e.py"

What it does:
- does a tighter local scan around an earlier proton winner
- can read either a previous best_candidate.json or summary.txt
- can also read the neutron anchor JSON
- when the anchor is present, the summary should show:
    - sqk_length_scale_source = neutron-anchor
      instead of
    - proton-fallback

............
- Written by Brendan Darrer aided by ChatGPT 5.4 (extensive thinking)
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

Install (generally):
    pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+
............

"The best next step is to make a tighter local scan around this winner and add the neutron-anchor workflow, so the length calibration stops being tautological." Yes, do this please!


STEP-BY-STEP COMMENTARY
=======================

What this script is for
-----------------------
This script performs a tighter local proton-like batch search around an earlier
winner. It is meant to be the second stage of the workflow:

1. First find a neutral / neutron-like anchor.
2. Then refine a proton-like charged candidate near a promising parameter set.

Why this matters
----------------
The neutron-like run gives a length-scale anchor in metres per simulation unit.
This proton script can then use that neutron anchor instead of forcing the
length scale from the proton itself. That avoids the earlier proton-fallback
situation where the proton Compton match was true by construction.

High-level flow of the script
-----------------------------
1. Read a previous best candidate JSON or summary.txt using --refine-from.
2. Build a smaller local parameter grid around that previous winner.
3. Optionally load a neutron anchor JSON file.
4. Run the same 1D shifted Model G solver for each local trial.
5. Score each trial in two ways:
   - Kelly proton profile score
   - SQK proxy-calibration score
6. Combine the scores if requested and rank the candidates.
7. Write out CSV tables, the best candidate JSON, SQK proxy JSON, plot, and
   summary text.

What the most important outputs mean
------------------------------------
- score_total:
    Original Kelly proton-shape score. Lower is better.
- score_sqk_proxy:
    Extra SQK-style proxy score. Lower is better.
- score_total_combined:
    Weighted blend of Kelly score and SQK proxy score.
- rank_score:
    The actual quantity used for ranking, depending on --rank-by.
- sqk_length_scale_source:
    Should ideally say neutron-anchor. If it says proton-fallback, the proton
    script had to force its own length calibration instead.

Important caution
-----------------
This script still works with the shifted pG, pX, pY solver variables and proxy
charge/mass estimates. So it is a practical refinement and ranking tool rather
than a final first-principles SI derivation.

""""
from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
import sys
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model_g_particle_1d_proton_batch_search_1c import (
    GridParams,
    ModelG1D,
    ModelParams,
    SeedParams,
    HAVE_SQK_MODULE,
    G_Ep_kelly,
    MASS_P,
    compute_sqk_proxy_metrics,
    make_summary_plot,
    parse_float_list,
    rho_from_GE_kelly,
    score_against_proton_target,
    write_csv,
)

PARAM_KEYS = ['dy', 'b', 'g', 'amp', 'sx', 'st', 'Tseed']


def parse_best_source(path: str) -> dict:
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' not in line:
                continue
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('Best candidate'):
                continue
            if line.startswith('Interpretation'):
                break
            if line.startswith('  '):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    data[key] = ast.literal_eval(value)
                except Exception:
                    data[key] = value
    return data


def load_neutron_anchor(path: str | None) -> float | None:
    if not path:
        return None
    payload = parse_best_source(path)
    for key in ('neutron_length_scale_m_per_unit', 'sqk_length_scale_m_per_unit'):
        val = payload.get(key)
        if val is not None:
            try:
                valf = float(val)
            except Exception:
                continue
            if np.isfinite(valf) and valf > 0.0:
                return valf
    return None


def build_local_axis(center: float, halfspan: float, npts: int, *, min_positive: float | None = None) -> list[float]:
    if npts <= 1 or halfspan <= 0.0:
        vals = np.asarray([center], dtype=float)
    else:
        vals = np.linspace(center - halfspan, center + halfspan, int(npts), dtype=float)
    if min_positive is not None:
        vals = np.maximum(vals, min_positive)
    vals = np.unique(np.round(vals, 12))
    return [float(v) for v in vals]


def resolve_scan_lists(args) -> tuple[dict, str | None]:
    if not args.refine_from:
        return {
            'dy': parse_float_list(args.dy),
            'b': parse_float_list(args.b),
            'g': parse_float_list(args.g),
            'amp': parse_float_list(args.amp),
            'sx': parse_float_list(args.sx),
            'st': parse_float_list(args.st),
            'Tseed': parse_float_list(args.Tseed),
        }, None

    src = parse_best_source(args.refine_from)
    missing = [k for k in PARAM_KEYS if k not in src]
    if missing:
        raise ValueError(f'refine source missing keys: {missing}')

    scan = {
        'dy': build_local_axis(float(src['dy']), args.dy_halfspan, args.refine_points, min_positive=1e-9),
        'b': build_local_axis(float(src['b']), args.b_halfspan, args.refine_points, min_positive=1e-9),
        'g': build_local_axis(float(src['g']), args.g_halfspan, args.refine_points, min_positive=1e-9),
        'amp': build_local_axis(float(src['amp']), args.amp_halfspan, args.refine_points, min_positive=1e-9),
        'sx': build_local_axis(float(src['sx']), args.sx_halfspan, args.refine_points, min_positive=1e-9),
        'st': build_local_axis(float(src['st']), args.st_halfspan, args.refine_points, min_positive=1e-9),
        'Tseed': build_local_axis(float(src['Tseed']), args.tseed_halfspan, args.refine_points, min_positive=0.0),
    }
    return scan, args.refine_from


def main() -> None:
    parser = argparse.ArgumentParser(description='Refined proton-like 1D Model G batch search with optional neutron anchor')
    parser.add_argument('--outdir', default='./model_g_proton_batch_search_out_1e', help='Output folder')
    parser.add_argument('--L', type=float, default=20.0)
    parser.add_argument('--nx', type=int, default=61)
    parser.add_argument('--tfinal', type=float, default=10.0)
    parser.add_argument('--max-step', type=float, default=0.08)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--nframes', type=int, default=40)
    parser.add_argument('--sign', type=int, default=-1)
    parser.add_argument('--a', type=float, default=14.0)
    parser.add_argument('--dx', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--s', type=float, default=0.0)
    parser.add_argument('--u', type=float, default=0.0)
    parser.add_argument('--v', type=float, default=0.0)
    parser.add_argument('--w', type=float, default=0.0)
    parser.add_argument('--dy', default='9.5,10.5,12.0', help='Comma list for manual scan')
    parser.add_argument('--b', default='27,28,29', help='Comma list for manual scan')
    parser.add_argument('--g', default='0.09,0.10,0.11', help='Comma list for manual scan')
    parser.add_argument('--amp', default='0.9,1.0,1.1', help='Comma list for manual scan')
    parser.add_argument('--sx', default='0.9,1.0,1.1', help='Comma list for manual scan')
    parser.add_argument('--st', default='1.5', help='Comma list for manual scan')
    parser.add_argument('--Tseed', default='3.0', help='Comma list for manual scan')
    parser.add_argument('--refine-from', default=None, help='Path to previous best_candidate.json or summary.txt for an automatic local scan')
    parser.add_argument('--refine-points', type=int, default=5, help='Points per parameter for automatic local scan')
    parser.add_argument('--dy-halfspan', type=float, default=0.75)
    parser.add_argument('--b-halfspan', type=float, default=1.0)
    parser.add_argument('--g-halfspan', type=float, default=0.01)
    parser.add_argument('--amp-halfspan', type=float, default=0.15)
    parser.add_argument('--sx-halfspan', type=float, default=0.15)
    parser.add_argument('--st-halfspan', type=float, default=0.30)
    parser.add_argument('--tseed-halfspan', type=float, default=0.60)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--rank-by', default='combined', choices=['kelly', 'sqk', 'combined'])
    parser.add_argument('--sqk-weight', type=float, default=0.35)
    parser.add_argument('--neutron-anchor-json', default=None,
                        help='Path to best_candidate_neutron_anchor.json or summary.txt from the neutron search')
    parser.add_argument('--neutron-length-scale-m-per-unit', type=float, default=None,
                        help='Optional explicit neutron anchor in metres per simulation unit')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    scan, refine_source = resolve_scan_lists(args)
    neutron_anchor = args.neutron_length_scale_m_per_unit
    if neutron_anchor is None:
        neutron_anchor = load_neutron_anchor(args.neutron_anchor_json)

    gp = GridParams(L=args.L, nx=args.nx, Tfinal=args.tfinal, max_step=args.max_step, rtol=args.rtol, atol=args.atol, dense=False)
    x_ref = np.linspace(-gp.L / 2.0, gp.L / 2.0, int(gp.nx))
    target_rho = rho_from_GE_kelly(np.abs(x_ref), G_Ep_kelly, MASS_P)

    rows = []
    best = None
    best_pY = None
    combos = [(dy, b, g, amp, sx, st, Tseed)
              for dy in scan['dy'] for b in scan['b'] for g in scan['g'] for amp in scan['amp']
              for sx in scan['sx'] for st in scan['st'] for Tseed in scan['Tseed']]
    t0 = time.time()

    for dy, b, g, amp, sx, st, Tseed in combos:
        mp = ModelParams(a=args.a, b=b, dx=args.dx, dy=dy, p=args.p, q=args.q, g=g, s=args.s, u=args.u, v=args.v, w=args.w)
        sp = SeedParams(sign=args.sign, amp=amp, sx=sx, st=st, Tseed=Tseed, nseeds=1)
        row = {
            'dy': dy, 'b': b, 'g': g, 'amp': amp, 'sx': sx, 'st': st, 'Tseed': Tseed,
            'solver_success': False, 'solver_message': '',
            'score_sqk_proxy': float('nan'), 'score_total_combined': float('nan')
        }
        try:
            model = ModelG1D(mp, gp, sp)
            sol = model.run(nframes=args.nframes)
            pG, pX, pY = model.unpack(sol.y[:, -1])
            diag = model.diagnostics(sol.y[:, -1])
            score = score_against_proton_target(model.x, pY, target_rho)
            sqk_proxy = compute_sqk_proxy_metrics(
                model, sol, pG, pX, pY,
                neutron_length_scale_m_per_unit=neutron_anchor,
            )
            row.update(diag)
            row.update(score)
            row.update(sqk_proxy)
            sqk_weight = min(max(float(args.sqk_weight), 0.0), 1.0)
            row['score_total_combined'] = float((1.0 - sqk_weight) * row['score_total'] + sqk_weight * row['score_sqk_proxy'])
            row.update({
                'solver_success': bool(sol.success),
                'solver_message': str(sol.message),
                'nfev': int(getattr(sol, 'nfev', -1)),
                'njev': int(getattr(sol, 'njev', -1)),
                'nlu': int(getattr(sol, 'nlu', -1)),
            })
            if args.rank_by == 'kelly':
                rank_score = row['score_total']
            elif args.rank_by == 'sqk':
                rank_score = row['score_sqk_proxy']
            else:
                rank_score = row['score_total_combined']
            row['rank_score'] = float(rank_score)
            if best is None or row['rank_score'] < best['rank_score']:
                best = row.copy()
                best_pY = pY.copy()
        except Exception as exc:
            row.update({
                'score_total': 1e9,
                'score_rho_rmse': 1e9,
                'score_surface_rmse': 1e9,
                'score_sqk_proxy': 1e9,
                'score_total_combined': 1e9,
                'rank_score': 1e9,
                'solver_success': False,
                'solver_message': repr(exc),
            })
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda rr: float(rr.get('rank_score', 1e9)))
    top_rows = rows_sorted[: max(1, min(args.topk, len(rows_sorted)))]

    all_csv = os.path.join(args.outdir, 'all_candidates.csv')
    top_csv = os.path.join(args.outdir, 'top_candidates.csv')
    best_json = os.path.join(args.outdir, 'best_candidate.json')
    best_sqk_json = os.path.join(args.outdir, 'best_candidate_sqk_proxy.json')
    plot_png = os.path.join(args.outdir, 'best_candidate_vs_kelly_proton.png')
    best_npz = os.path.join(args.outdir, 'best_candidate_profiles.npz')
    summary_txt = os.path.join(args.outdir, 'summary.txt')

    preferred = [
        'rank_score', 'score_total_combined', 'score_total', 'score_sqk_proxy', 'score_rho_rmse', 'score_surface_rmse',
        'dy', 'b', 'g', 'amp', 'sx', 'st', 'Tseed',
        'pG_core', 'pX_core', 'pY_core', 'Qproxy_int_pYdx', 'pY_peak_abs', 'pY_fwhm_abs', 'polarity_label',
        'penalty_core_sign', 'penalty_charge_sign', 'penalty_negative_lobes', 'penalty_fwhm_rel',
        'sqk_lambda_sim', 'sqk_r_core_sim', 'sqk_shell_spacing_error', 'sqk_core_bias_x', 'sqk_core_bias_y',
        'sqk_outer_bias_x', 'sqk_outer_bias_y', 'sqk_q_core_proxy', 'sqk_q_abs_core_proxy', 'sqk_sg_core_proxy',
        'sqk_q_far_proxy', 'sqk_charge_consistency', 'sqk_tail_rel_error', 'sqk_proton_charge_bias_error',
        'sqk_stability_error', 'sqk_length_scale_m_per_unit', 'sqk_length_scale_source',
        'sqk_proton_lambda_m', 'sqk_proton_lambda_error_frac', 'sqk_charge_scale_c_per_proxy',
        'sqk_active_mass_scale_kg_per_proxy', 'solver_success', 'solver_message', 'nfev', 'njev', 'nlu'
    ]
    extra = sorted({k for row in rows_sorted for k in row.keys()} - set(preferred))
    fieldnames = preferred + extra
    write_csv(all_csv, [{k: row.get(k, '') for k in fieldnames} for row in rows_sorted], fieldnames)
    write_csv(top_csv, [{k: row.get(k, '') for k in fieldnames} for row in top_rows], fieldnames)

    if best is not None and best_pY is not None:
        with open(best_json, 'w', encoding='utf-8') as jf:
            json.dump(best, jf, indent=2)
        with open(best_sqk_json, 'w', encoding='utf-8') as jf:
            json.dump({k: v for k, v in best.items() if k.startswith('sqk_') or k in {'rank_score', 'score_total_combined', 'score_total', 'score_sqk_proxy'}}, jf, indent=2)
        np.savez(best_npz, x=x_ref, pY=best_pY, target_rho=target_rho)
        make_summary_plot(plot_png, x_ref, target_rho, best_pY, best)

    elapsed = time.time() - t0
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('Model G 1D proton-like batch search summary\n')
        f.write(f'trials = {len(rows)}\n')
        f.write(f'elapsed_sec = {elapsed:.3f}\n')
        f.write(f'rank_by = {args.rank_by}\n')
        f.write(f'sqk_weight = {args.sqk_weight}\n')
        f.write(f'sqk_module_available = {HAVE_SQK_MODULE}\n')
        f.write(f'refine_source = {refine_source}\n')
        f.write(f'neutron_anchor_m_per_unit = {neutron_anchor}\n')
        if best is not None:
            f.write('Best candidate\n')
            for key in fieldnames:
                if key in best:
                    f.write(f'  {key} = {best[key]}\n')
            f.write('\nInterpretation\n')
            f.write('  score_total is the Kelly proton shape score; score_sqk_proxy is the SQK proxy-calibration score.\n')
            f.write('  When neutron_anchor_m_per_unit is present, sqk_length_scale_source should read neutron-anchor instead of proton-fallback.\n')
            f.write('  refine_source records the earlier winner used to build the tighter local scan.\n')

    print(f'Wrote: {all_csv}')
    print(f'Wrote: {top_csv}')
    print(f'Wrote: {best_json}')
    print(f'Wrote: {best_sqk_json}')
    print(f'Wrote: {best_npz}')
    print(f'Wrote: {plot_png}')
    print(f'Wrote: {summary_txt}')


if __name__ == '__main__':
    main()
