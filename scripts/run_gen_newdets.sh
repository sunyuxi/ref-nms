python -m tools.save_matt_dets --dataset rsvg --m att_vanilla --tid refnms256 --conf 0.045 \
            --refnmsdet_jsonpath refnms_det_instances_rsvg.json \
            --old_refnmsdet_dirpath hbb_obb_features_refnms_det \
            --new_refnmsdet_dirpath hbb_obb_features_refnms_det_selected256 \
            --refnmsdet_feats_suffix hbb_det_res50_dota_v1_0_RoITransformer.hdf5
#changed: tid, new_refnmsdet_dirpath
#output/matt_dets_att_vanilla_refnms256_rsvg_0.json
