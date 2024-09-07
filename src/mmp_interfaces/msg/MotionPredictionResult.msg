### Motion prediction
std_msgs/Header header # header with frame_id and timestamp
mmp_interfaces/HumanTrajectoryArray[] mu_list_list      # [t1_list, t2_list, ...], t1_list = [mu1, mu2, ...],       mu1 = [x, y]
mmp_interfaces/HumanTrajectoryArray[] std_list_list     # [t1_list, t2_list, ...], t1_list = [std1, std2, ...],     std1 = [xx, yy]
mmp_interfaces/HumanTrajectoryArray[] conf_list_list    # [t1_list, t2_list, ...], t1_list = [conf1, conf2, ...],   conf1 = [c, c]
