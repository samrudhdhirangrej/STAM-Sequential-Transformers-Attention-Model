python3 evaluate.py \
        --batch_size_val=625 \
        --trainmaxT=21 \
        --maxT=49 \
        --num_workers=8 \
        --model_name='STAM_deit_small_patch16_224_fMoW' \
        --pin_mem=True \
        --drop=0.0 \
        --drop_path=0.1 \
        --checkpoint_filename='STAM_fMoW.pth' \
        --output_dir='fMoW' \
        --epochs=10 \
        --dataset='fMoW' \
        --pretrained=True \
        --input_size=224 \
        --sync_bn=True \
        --mlp_layers=4 \
        --mlp_hidden_dim=2048 \
