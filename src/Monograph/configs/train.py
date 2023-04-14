# # training with hypersim config
# config = {
#     'input_path':           "/mnt/datasets/hypersim_graphs/",
#     'learning_rate':        0.005,
#     'epochs':               50,
#     'batch_size':           20,
#     'num_train':            20,
#     'num_test':             10,
#     'scheduler_step':       10,
#     'scheduler_gamma':      0.9,
#     'triplet_loss_margin':  0.25,
#     'triplet_loss_p':       2,
#     'data_source':          'pipeline',
#     'model_name':           'pipeline'
# }

# training with 3dssg config
config = {
    'input_path':           '/mnt/datasets/3dssg/',
    'learning_rate':        0.005,
    'epochs':               50,
    'batch_size':           20,
    'num_train':            20,
    'num_test':             10,
    'scheduler_step':       10,
    'scheduler_gamma':      0.9,
    'triplet_loss_margin':  0.25,
    'triplet_loss_p':       2,
    'data_source':          'ssg',
    'model_name':           'ssg'
}