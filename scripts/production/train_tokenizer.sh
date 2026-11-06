python main_gensid.py \
  --sid_model rqvae \
  --dataset Instruments \
  --device cuda:0 \
  --embeddings_path data/Instruments/Instruments.emb-llama-td.npy \
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt \
  --fix_collisions \
  --saved_model_dir tokenizer
