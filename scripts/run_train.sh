#nohup python -m tools.train_att_vanilla > logs/train.log 2>&1 &

nohup python -m tools.train_att_vanilla --dataset rsvg --gpu-id 0 --tid refnms256 \
        --refnmsdet_jsonpath refnms_det_instances_rsvg.json \
        --refnmsdet_dirpath hbb_obb_features_refnms_det \
        --refnmsdet_feats_suffix hbb_det_res50_dota_v1_0_RoITransformer.hdf5 > logs/train256.log 2>&1 &
