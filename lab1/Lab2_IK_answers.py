import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

def rotation_matrix(a, b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    n = np.cross(a, b)
    # 旋转矩阵是正交矩阵，矩阵的每一行每一列的模，都为1；并且任意两个列向量或者任意两个行向量都是正交的。
    # n=n/np.linalg.norm(n)
    # 计算夹角
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(n)
    theta = np.arctan2(sin_theta, cos_theta)
    # 构造旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    rotation_matrix = np.array([[n[0]*n[0]*v+c, n[0]*n[1]*v-n[2]*s, n[0]*n[2]*v+n[1]*s],
                                 [n[0]*n[1]*v+n[2]*s, n[1]*n[1]*v+c, n[1]*n[2]*v-n[0]*s],
                                 [n[0]*n[2]*v-n[1]*s, n[1]*n[2]*v+n[0]*s, n[2]*n[2]*v+c]])
    return rotation_matrix

def inv_safe(data):
    # return R.from_quat(data).inv()
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return np.linalg.inv(R.from_quat(data).as_matrix())
    
def from_quat_safe(data):
    # return R.from_quat(data)
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return R.from_quat(data).as_matrix()
# class MetaData:
#     def __init__(self, joint_name, joint_parent, joint_initial_position, root_joint, end_joint):
#         """
#         一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
#         root_joint是固定节点的索引，并不是RootJoint节点
#         """
#         self.joint_name = joint_name
#         self.joint_parent = joint_parent
#         self.joint_initial_position = joint_initial_position
#         self.root_joint = root_joint
#         self.end_joint = end_joint

#     def get_path_from_root_to_end(self):
#         """
#         辅助函数，返回从root节点到end节点的路径
        
#         输出：
#             path: 各个关节的索引
#             path_name: 各个关节的名字
#         Note: 
#             如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
#             在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
#             你可能会需要这两个输出。
#         """
        
#         # 从end节点开始，一直往上找，直到找到腰部节点
#         path1 = [self.joint_name.index(self.end_joint)]
#         while self.joint_parent[path1[-1]] != -1:
#             path1.append(self.joint_parent[path1[-1]])
            
#         # 从root节点开始，一直往上找，直到找到腰部节点
#         path2 = [self.joint_name.index(self.root_joint)]
#         while self.joint_parent[path2[-1]] != -1:
#             path2.append(self.joint_parent[path2[-1]])
        
#         # 合并路径，消去重复的节点
#         while path1 and path2 and path2[-1] == path1[-1]:
#             path1.pop()
#             a = path2.pop()
            
#         path2.append(a)
#         path = path2 + list(reversed(path1))
#         path_name = [self.joint_name[i] for i in path]
#         return path, path_name, path1, path2


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    
    # joint_positions, joint_orientations = part1_inverse_kinematics_ccd(meta_data, joint_positions, joint_orientations, target_pose)
    joint_positions, joint_orientations = part1_inverse_kinematics_jacobian(meta_data, joint_positions, joint_orientations, target_pose)
    
    
    return joint_positions, joint_orientations
def part1_inverse_kinematics_ccd(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path,path_name,path1,path2=meta_data.get_path_from_root_to_end()
    parent_idx=meta_data.joint_parent
    # local_rotation是用于最后计算不在链上的节点
    no_caled_orientation=copy.deepcopy(joint_orientations)
    local_rotation = [
        R.from_matrix(inv_safe(joint_orientations[parent_idx[i]]) * from_quat_safe(joint_orientations[i])).as_quat() for i
                      in range(len(joint_orientations))]
    local_rotation[0] = R.from_matrix(from_quat_safe(joint_orientations[0])).as_quat()
    local_position = [joint_positions[i]-joint_positions[parent_idx[i]] for i
                      in range(len(joint_orientations))]
    local_position[0] = joint_positions[0]

    path_end_id=path1[0] ## lWrist_end 就是手掌 只是加了end不叫hand而已
    for k in range(0,300):
        # k：循环次数
        # 正向的，path1是从手到root之前
        for idx in range(0,len(path1)):
            # idx：路径上的第几个节点了，第0个是手，最后一个是root
            path_joint_id=path1[idx]

            vec_to_end=joint_positions[path_end_id]-joint_positions[path_joint_id]
            vec_to_target=target_pose-joint_positions[path_joint_id]
            # 获取end->target的旋转矩阵
            # debug
            # rot_matrix=rotation_matrix(np.array([1,0,0]),np.array([1,1,0]))
            rot_matrix=rotation_matrix(vec_to_end,vec_to_target)

            # 计算前的朝向。这个朝向实际上是累乘到父节点的
            initial_orientation=from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R=R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation=rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id]=R.from_matrix(calculated_orientation).as_quat()

            # 子节点的朝向也会有所变化
            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(idx-1,0,-1):
                path_joint_id=path1[i]
                # 遍历路径后的节点,都乘上旋转
                joint_orientations[path_joint_id]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            path_joint_id=path1[idx]
            # 修改子节点的位置
            for i in range(idx-1,-1,-1):
                # path_joint_id=path1[i]
                # 节点id
                next_joint_id=path1[i]
                # 指向下个节点的向量
                vec_to_next=joint_positions[next_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[next_joint_id]=calculated_vec_to_next+joint_positions[path_joint_id]


        for idx in range(len(path2)-1,0,-1): # len(path2)-1 --> 0
            path_joint_id=path2[idx]
            parient_joint_id=max(parent_idx[path_joint_id],0)

            vec_to_end=joint_positions[path_end_id]-joint_positions[path_joint_id]
            vec_to_target=target_pose-joint_positions[path_joint_id]
            # 获取end->target的旋转矩阵
            rot_matrix=rotation_matrix(vec_to_end,vec_to_target)
            # 计算前的朝向。注意path2是反方向的，要改父节点才行
            initial_orientation=from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R= R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation=rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id]=R.from_matrix(calculated_orientation).as_quat()

            # 其他节点的朝向也会有所变化
            for i in range(idx+1,len(path2)):
                path_joint_id=path2[i] 
                joint_orientations[path_joint_id]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(len(path1)-1,0,-1):
                path_joint_id=path1[i]
                # 遍历路径后的节点,都乘上旋转
                joint_orientations[path_joint_id]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            path_joint_id=path2[max(idx-1,0)]
            # 修改父节点，或者说更靠近手的那些节点的位置
            # path2上的
            for i in range(idx,len(path2)):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id=path2[i]
                # 指向上一个节点的向量
                vec_to_next=joint_positions[prev_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id]=joint_positions[path_joint_id]+calculated_vec_to_next
            # path1上的
            for i in range(len(path1)-1,-1,-1):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id=path1[i]
                # 指向上一个节点的向量
                vec_to_next=joint_positions[prev_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id]=calculated_vec_to_next+joint_positions[path_joint_id]

        # debug
        # rot_matrix=rotation_matrix(np.array([1,0,0]),np.array([1,0,1]))
        # joint_orientations[0]=R.from_matrix(rot_matrix).as_quat()
        # joint_orientations[1]=R.from_matrix(rot_matrix).as_quat()
        joint_orientations[path_end_id]=joint_orientations[path1[1]]
        cur_dis=np.linalg.norm(joint_positions[path_end_id]-target_pose)
        if cur_dis<0.01:
            break
    print("距离",cur_dis,"迭代了",k,"次")
    # 更新不在链上的节点
    for k in range(len(joint_orientations)):
        if k in path:
            pass
        elif k==0:
            # 要单独处理，不然跟节点的-1就会变成从最后一个节点开始算
            pass
        else:
            # 先获取局部旋转
            # 这里如果直接存的就是矩阵就会有问题？
            local_rot_matrix=R.from_quat(local_rotation[k]).as_matrix()
            # 再获取我们已经计算了的父节点的旋转
            parent_rot_matrix=from_quat_safe(joint_orientations[parent_idx[k]])
            # 乘起来
            # re=local_rot_matrix.dot(parent_rot_matrix)
            re=parent_rot_matrix.dot(local_rot_matrix)
            joint_orientations[k]=R.from_matrix(re).as_quat()

            # 父节点没旋转的时候是：
            initial_o=from_quat_safe(no_caled_orientation[parent_idx[k]])
            # 父节点的旋转*delta_orientation=子节点旋转
            # 反求delta_orientation
            delta_orientation = np.dot(re, np.linalg.inv(initial_o))
            # 父节点的位置加原本基础上的旋转
            joint_positions[k]=joint_positions[parent_idx[k]]+delta_orientation.dot(local_position[k])

    return joint_positions, joint_orientations

def part1_inverse_kinematics_jacobian(meta_data, joint_positions, joint_orientations, target_pose):
    """
    使用雅可比矩阵方法计算逆运动学，允许全身参与运动
    """
    # 获取关节链路径
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_parent = meta_data.joint_parent
    joint_initial_pos = meta_data.joint_initial_position

    def get_joint_rotations():
        """计算每个关节的局部旋转（相对于父关节）"""
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_orientations)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * 
                                     R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        """计算每个关节相对于父关节的偏移"""
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_positions)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_pos[i] - joint_initial_pos[joint_parent[i]]
        return joint_offsets

    # 获取关节的局部旋转和偏移
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()
    
    # 使用整个路径中的关节，包括上半身和下半身
    # 从路径中提取所有关节，除了根节点
    active_joints = path[1:]  # 包含上半身和下半身的所有关节
    
    # 获取末端节点索引
    end_joint_idx = meta_data.joint_name.index(meta_data.end_joint)
    
    # 优化参数
    max_iter = 300
    damping = 1  # 阻尼因子，用于LM算法
    step_size = 0.2
    tolerance = 1e-2
    
    # 备份原始位置和方向
    original_positions = copy.deepcopy(joint_positions)
    original_orientations = copy.deepcopy(joint_orientations)
    
    # 迭代优化
    for iter_count in range(max_iter):
        # 当前末端位置
        end_position = joint_positions[end_joint_idx]
        
        # 计算当前误差
        error = target_pose - end_position
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude < tolerance:
            print(f"IK 收敛，迭代 {iter_count} 次，误差：{error_magnitude:.6f}")
            break
            
        # 构建雅可比矩阵
        jacobian = np.zeros((3, 3 * len(active_joints)))
        
        for i, joint_idx in enumerate(active_joints):
            joint_pos = joint_positions[joint_idx]
            # 计算旋转轴（三个自由度）
            joint_orient = R.from_quat(joint_orientations[joint_idx])
            
            # 获取全局坐标系中的轴向量
            axis_x = joint_orient.apply([1, 0, 0])
            axis_y = joint_orient.apply([0, 1, 0])
            axis_z = joint_orient.apply([0, 0, 1])
            
            # 当前关节到末端的向量
            r = end_position - joint_pos
            
            # 计算每个轴的影响
            col_x = np.cross(axis_x, r)
            col_y = np.cross(axis_y, r)
            col_z = np.cross(axis_z, r)
            
            # 填充雅可比矩阵
            jacobian[:, 3*i] = col_x
            jacobian[:, 3*i+1] = col_y
            jacobian[:, 3*i+2] = col_z
        
        # 使用阻尼最小二乘法
        JtJ = jacobian.T @ jacobian
        lambda_I = damping * np.eye(JtJ.shape[0])
        delta_theta = np.linalg.solve(JtJ + lambda_I, jacobian.T @ error) * step_size
        
        # 更新关节角度
        for i, joint_idx in enumerate(active_joints):
            delta_rot = R.from_euler('XYZ', delta_theta[3*i:3*i+3])
            current_rot = R.from_quat(joint_orientations[joint_idx])
            # 更新旋转
            new_rot = delta_rot * current_rot
            joint_orientations[joint_idx] = new_rot.as_quat()
        
        # 正向运动学：更新所有关节位置
        for joint_idx in range(len(joint_positions)):
            if joint_idx == 0 or joint_parent[joint_idx] == -1:
                continue  # 跳过根节点
                
            parent_idx = joint_parent[joint_idx]
            parent_pos = joint_positions[parent_idx]
            parent_orient = R.from_quat(joint_orientations[parent_idx])
            
            # 计算初始偏移
            offset = joint_initial_pos[joint_idx] - joint_initial_pos[parent_idx]
            # 应用父节点旋转
            rotated_offset = parent_orient.apply(offset)
            # 更新位置
            joint_positions[joint_idx] = parent_pos + rotated_offset
    
    if iter_count == max_iter - 1:
        print(f"IK 未收敛，最终误差：{error_magnitude:.6f}")
    
    return joint_positions, joint_orientations



def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations