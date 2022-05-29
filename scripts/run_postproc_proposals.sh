
python -m tools.save_ref_nms_proposals --gpu-id 0 --dataset rsvg --tid refnms256 --m att_vanilla \
    --refnmsdet_jsonpath refnms_det_instances_rsvg.json \
    --refnmsdet_dirpath hbb_obb_features_refnms_det \
    --refnmsdet_feats_suffix hbb_det_res50_dota_v1_0_RoITransformer.hdf5
#changed:tid
#cache/proposals_att_vanilla_rsvg_refnms.pkl
