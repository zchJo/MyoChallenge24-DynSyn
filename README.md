# MyoChallenge-2024-DynSyn

## Working directory

- myosuite (gitignore)
  - git clone git@github.com:MyoHub/myosuite.git --recursive
  - pip install -e . in the conda env
  - To check the original challenge environment
  - Add self.render_mode = 'rgb_array' in env_base.py, line 64
  - Change color = "white" in prompt_utils.py, line 78
- myochallenge_2024eval
  - git clone git@github.com:MyoHub/myochallenge_2024eval.git
  - To build local submissions, which is faster
  - Add evalai token
      - https://github.com/MyoHub/myochallenge_2024eval/blob/main/tutorials/DIY_Submission.md
- challenge_wrapper
  - To do reward shaping in challenge environments
- DynSyn
  - git clone git@github.com:Beanpow/DynSyn.git
  - DynSyn and its training code
  - configs
  - log (gitignore)


## Tasks

### Locomotion

https://myosuite.readthedocs.io/en/latest/challenge-doc.html#prosthesis-locomotion

action_space.shape = (54, ), low = -1, high = 1

Control only the muscles of the leg model, but not prostheic motors.

#### Leg model muscle groups

- Motor Number = 2
  - Knee
  - Ankle

Muscle Number = 14 + 40 = 54

Synergy List, Group Number = 28 + 2
[[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 13], [12], [14, 15, 19], [16, 17], [18], [20, 21], [22], [23], [24], [25], [26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36], [37, 45, 46], [38, 42], [39, 40], [41], [43], [44, 48], [47], [49], [50], [51, 52, 53]]

Name List
[['addbrev_r', 'addlong_r'], ['glmax1_r', 'glmax2_r', 'glmax3_r'], ['glmed1_r', 'glmed2_r', 'glmed3_r'], ['glmin1_r', 'glmin2_r', 'glmin3_r'], ['iliacus_r', 'psoas_r'], ['piri_r'], ['addbrev_l', 'addlong_l', 'addmagProx_l'], ['addmagDist_l', 'addmagIsch_l'], ['addmagMid_l'], ['bflh_l', 'bfsh_l'], ['edl_l'], ['ehl_l'], ['fdl_l'], ['fhl_l'], ['gaslat_l', 'gasmed_l'], ['glmax1_l', 'glmax2_l', 'glmax3_l'], ['glmed1_l', 'glmed2_l', 'glmed3_l'], ['glmin1_l', 'glmin2_l', 'glmin3_l'], ['grac_l', 'semimem_l', 'semiten_l'], ['iliacus_l', 'psoas_l'], ['perbrev_l', 'perlong_l'], ['piri_l'], ['recfem_l'], ['sart_l', 'tfl_l'], ['soleus_l'], ['tibant_l'], ['tibpost_l'], ['vasint_l', 'vaslat_l', 'vasmed_l']]

['addbrev_r', 'addlong_r']
['glmax1_r', 'glmax2_r', 'glmax3_r']
['glmed1_r', 'glmed2_r', 'glmed3_r']
['glmin1_r', 'glmin2_r', 'glmin3_r']
['iliacus_r', 'psoas_r']
['piri_r']

['addbrev_l', 'addlong_l', 'addmagProx_l']
['addmagDist_l', 'addmagIsch_l']
['addmagMid_l']
['bflh_l', 'bfsh_l']
['edl_l']
['ehl_l']
['fdl_l']
['fhl_l']
['gaslat_l', 'gasmed_l']
['glmax1_l', 'glmax2_l', 'glmax3_l']
['glmed1_l', 'glmed2_l', 'glmed3_l']
['glmin1_l', 'glmin2_l', 'glmin3_l']
['grac_l', 'semimem_l', 'semiten_l']
['iliacus_l', 'psoas_l']
['perbrev_l', 'perlong_l']
['piri_l']
['recfem_l']
['sart_l', 'tfl_l']
['soleus_l']
['tibant_l']
['tibpost_l']
['vasint_l', 'vaslat_l', 'vasmed_l']

### Manipulation

https://myosuite.readthedocs.io/en/latest/challenge-doc.html#prosthesis-co-manipulation

action_space.shape = (80, ), low = -1, high = 1

Control the muscles of the arm model, and the prothetic motors.

#### Arm model muscle groups

- Position Motor Number = 17
  - Arm: 4
  - Wrist: 3
  - Thumb: 4
  - Fingers: 6

Muscle Number = 63

Synergy List, Group Number = 33 + 17
[[0], [1], [2], [3], [4, 6], [5], [7], [8], [9, 10, 14], [11, 12, 13], [15, 16, 17, 18], [19], [20, 21], [22, 23], [24, 25], [26], [27, 29], [28], [30], [31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44], [45], [46], [47, 49], [48], [50], [51, 52, 53], [54, 55, 56], [57, 58, 59], [60, 61, 62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79]]

Name List
[['DELT1'], ['DELT2'], ['DELT3'], ['SUPSP'], ['INFSP', 'TMIN'], ['SUBSC'], ['TMAJ'], ['PECM1'], ['PECM2', 'PECM3', 'CORB'], ['LAT1', 'LAT2', 'LAT3'], ['TRIlong', 'TRIlat', 'TRImed', 'ANC'], ['SUP'], ['BIClong', 'BICshort'], ['BRA', 'BRD'], ['ECRL', 'ECRB'], ['ECU'], ['FCR', 'PL'], ['FCU'], ['PT'], ['PQ'], ['FDS5', 'FDS4', 'FDS3', 'FDS2'], ['FDP5', 'FDP4', 'FDP3', 'FDP2'], ['EDC5', 'EDC4', 'EDC3', 'EDC2'], ['EDM'], ['EIP'], ['EPL'], ['EPB', 'APL'], ['FPL'], ['OP'], ['RI2', 'LU_RB2', 'UI_UB2'], ['RI3', 'LU_RB3', 'UI_UB3'], ['RI4', 'LU_RB4', 'UI_UB4'], ['RI5', 'LU_RB5', 'UI_UB5']]

['DELT1']
['DELT2']
['DELT3']
['SUPSP']
['INFSP', 'TMIN']
['SUBSC']
['TMAJ']
['PECM1']
['PECM2', 'PECM3', 'CORB']
['LAT1', 'LAT2', 'LAT3']
['TRIlong', 'TRIlat', 'TRImed', 'ANC']
['SUP']
['BIClong', 'BICshort']
['BRA', 'BRD']
['ECRL', 'ECRB']
['ECU']
['FCR', 'PL']
['FCU']
['PT']
['PQ']
['FDS5', 'FDS4', 'FDS3', 'FDS2']
['FDP5', 'FDP4', 'FDP3', 'FDP2']
['EDC5', 'EDC4', 'EDC3', 'EDC2']
['EDM']
['EIP']
['EPL']
['EPB', 'APL']
['FPL']
['OP']
['RI2', 'LU_RB2', 'UI_UB2']
['RI3', 'LU_RB3', 'UI_UB3']
['RI4', 'LU_RB4', 'UI_UB4']
['RI5', 'LU_RB5', 'UI_UB5']