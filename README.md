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
