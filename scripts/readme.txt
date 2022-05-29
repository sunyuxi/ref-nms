1、data文件夹
data/glove.840B.300d.txt
data/refer/rsvg，rsvg替换成具体的数据集
其中rsvg文件夹需要包含以下数据文件:
images
refs_rsvg.p
instances_rsvg.json
refnms_det_instances_rsvg.json # 所有检测框的json数据
hbb_obb_features_refnms_det #存储每个检测框对应的特征的文件夹
hbb_det_res50_dota_v1_0_RoITransformer.hdf5 #每个检测框对应特征的文件名的后缀

2、执行prepare_data.sh，生成std_vocab_rsvg.txt,std_glove_rsvg.npy,std_refdb_rsvg.json,std_ctxdb_rsvg.json
tools/build_vocab.py: std_vocab_rsvg.txt,std_glove_rsvg.npy，即根据REFER类，生成对应的词典表示和向量
tools/build_refdb.py: std_refdb_rsvg.json，根据REFER生成refer（sentid,refid,annid）的json表示
tools/build_ctxdb.py: std_ctxdb_rsvg.json，根据REFER数据，利用类别的词向量，将类别向量相似的object box视为context

3、如果更换数据集，lib/predictor.py中AttVanillaPredictorV2需要修改特征维度
4、执行run_train.sh训练模型，训练模型的文件，会存储到output文件夹中
5、执行run_postproc_proposals.sh，每个proposal(bbox)生成新的分数，生成文件为：'cache/proposals_att_vanilla_rsvg_refnms256.pkl'
6、执行run_gen_newdets.sh，生成MAttNet，NMTree所需的json文件，生成文件为：output/matt_dets_att_vanilla_refnms256_rsvg_0.json

最后生成的json文件，相当于MAttNet中tools/run_detect.py生成的json文件
