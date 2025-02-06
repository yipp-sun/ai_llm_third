from datasets import load_dataset, load_from_disk

traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)

traindata.save_to_disk('ptb_train')
valdata.save_to_disk('ptb_val')
