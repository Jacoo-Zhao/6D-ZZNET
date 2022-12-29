### 6D-ZZNET

[**CODALAB Chanllenge**: No GPS no problem! Democratising aerial navigation via robust and data-scalable computer vision.](https://codalab.lisn.upsaclay.fr/competitions/5481#learn_the_details-overview)

![Chanllenge](https://lh6.googleusercontent.com/DhS-jaBBTRywlgF9fjzA3lDhKPiQ8nAQ4ELrxySrMX9qObxiN889t1gvw7Cg5gnFfnTi6D9XrAQs0Fh7wsew3Mbv8y7lhHyTPMgVSO7oy6Jje9itEPen5OtTrAers_ALz-Szg6NDRKtskDzcSA "Challenge")

Data processing pipeline:
    source: 1. train/train/rdb; poses
            2. train/synth_real_matching/rgb_real+rgb_synth; poses+poses_copy     
        
    poses.csv:poses, file offered.
    tuple.pickle: ground_truth, poses.csv, imgs; function: tuple_formulate in utilis.py
    dataloader: imgs, tuple.pickle

1. mv all imgs and poses in same. folder
2. code different folder. as [list] ☑

poses.csv:done
tuple.pickle: in process 