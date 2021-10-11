python encoder.py --policy data/attt/policies/p1_policy2.txt --states data/attt/states/states_file_p2.txt > mdp_p1_ag_p2_pol2
python planner.py --mdp mdp_p1_ag_p2_pol2 > polval_p1_ag_p2_pol2
python decoder.py --value-policy polval_p1_ag_p2_pol2 --states data/attt/states/states_file_p2.txt --player-id 2 > pol_p1_ag_p2_pol2
python simul.py -p2 pol_p1_ag_p2_pol2 -p1 data/attt/policies/p1_policy2.txt
rm polval_p1_ag_p2_pol2
rm pol_p1_ag_p2_pol2
rm mdp_p1_ag_p2_pol2
