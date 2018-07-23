#python train.py --model  models.net_pinch.VGG16 --dataroot ../data/artificial/train    --dataroot_val ../data/artificial/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2 
#python train.py --model  models.net_pinch_resnet.VGG16 --dataroot ../data/artificial/train    --dataroot_val ../data/artificial/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2 
#python train.py --model  models.net_pinch_googlenet.VGG16 --dataroot ../data/artificial/train    --dataroot_val ../data/artificial/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2 
#python train.py --model  models.net_pinch_vgg.VGG16 --dataroot ../data/artificial/train    --dataroot_val ../data/artificial/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2 
#python train.py --model  models.net_pinch_vgg_lambda.VGG16 --w_lambda 1.0 --dataroot ../data/artificial/train    --dataroot_val ../data/artificial/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2 
#python train.py --model  models.net_pinch_vgg_lambda.VGG16 --w_lambda 1.0 --save_epoch_freq 3 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2
#python train.py --model  models.net_pinch_vgg_lambda_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2
#python train.py --model  models.net_pinch_vgg_lambda_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2
#python train.py --model  models.net_pinch_vgg_lambda_crop_small.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2
#python train.py --model  models.net_pinch_resnet_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2
#python train.py --model  models.net_pinch_densenet_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 256 --checkpoints_dir $1     --gpu_ids $2


 
#python train.py --num_channels 1 --face_lambda_len 2 --model  models.net_pinch_vgg_lambda_crop_5.VGG16 --w_lambda 1.0 --save_epoch_freq 100 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 224 --checkpoints_dir $1     --gpu_ids $2

#python train.py --num_channels 1 --face_lambda_len 2 --model  models.net_pinch_vgg_lambda_crop_3.VGG16 --w_lambda 1.0 --save_epoch_freq 100 --dataroot ../data/0704/cls/artificial_female/20001/train    --dataroot_val ../data/0704/cls/artificial_female/20001/test   --batch_size 48  --load_size 224 --checkpoints_dir $1     --gpu_ids $2

#python train.py --num_channels 1 --face_lambda_len 2 --model  models.net_pinch_vgg_lambda_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 100 --dataroot ../data/0706/artificial_nose/train    --dataroot_val ../data/0706/artificial_nose/test   --batch_size 48  --load_size 224 --checkpoints_dir $1     --gpu_ids $2
#python train.py --num_channels 1 --face_lambda_len 2 --model  models.net_pinch_resnet_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 100 --dataroot ../data/0706/artificial_nose/train  --dataroot_val ../data/0706/artificial_nose/test   --batch_size 48  --load_size 224  --checkpoints_dir $1     --gpu_ids $2

#python train.py --num_channels 3 --face_lambda_len 27 --model  models.net_pinch_vgg_lambda_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../../data/0711/dataset_wholeface/data/train  --dataroot_val ../../data/0711/dataset_wholeface/data/test   --batch_size 48  --load_size 224  --checkpoints_dir $1     --gpu_ids $2
python train.py --num_channels 3 --face_lambda_len 27 --model  models.net_pinch_vgg_lambda_crop.VGG16 --w_lambda 1.0 --save_epoch_freq 10 --dataroot ../../data/0712/data/train  --dataroot_val ../../data/0712/data/test   --batch_size 48  --load_size 224  --checkpoints_dir $1     --gpu_ids $2