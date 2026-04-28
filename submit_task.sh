#!/bin/bash
#SBATCH -o /lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/logs/job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan      # 作业提交的指定分区队列为titan
#SBATCH --qos=titan           # 指定作业的QOS
#SBATCH -J myFirstGPUJob       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 
#SBATCH --time=72:00:00

echo "Multi titan train on max 500"

source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh

conda activate msg3d

python3 main.py --config /lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/config/kinetics-skeleton/train_joint_500.yaml --work-dir /lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/checkpoints/self-kinetics-joint-500 --device 0 1 --half