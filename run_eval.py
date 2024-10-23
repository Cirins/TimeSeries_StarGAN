import subprocess


dataset = 'realworld' # 'realworld' or 'cwru'

if dataset == 'realworld':
    dataset_name = 'realworld_128_3ch_4cl'
    class_names = ['WAL', 'RUN', 'CLD', 'CLU']
    channel_names = ['X', 'Y', 'Z']

    num_timesteps = 128
    num_channels = 3
    num_train_domains = 10
    num_test_domains = 5
    num_classes = 4

    lambda_cyc = 5
    lambda_id = 5
    lambda_dom = 0.1
    lambda_ds = 1

    total_iters = 40000

elif dataset == 'cwru':
    dataset_name = 'cwru_256_3ch_5cl'
    class_names = ['IR', 'Ball', 'OR_centred', 'OR_orthogonal', 'OR_opposite']
    channel_names = ['DE', 'FE', 'BA']

    num_timesteps = 256
    num_channels = 3
    num_train_domains = 4
    num_test_domains = 4
    num_classes = 5

    lambda_cyc = 50
    lambda_id = 50
    lambda_dom = 0.01
    lambda_ds = 1

    total_iters = 80000


# Launch evaluation
print('\n\nStarting evaluation phase...\n\n')
subprocess.run(['python', 'main.py',
                '--mode', 'eval',
                '--dataset', dataset,
                '--dataset_name', dataset_name,
                '--class_names', ','.join(class_names),
                '--channel_names', ','.join(channel_names),
                '--num_timesteps', str(num_timesteps),
                '--num_channels', str(num_channels),
                '--num_train_domains', str(num_train_domains),
                '--num_test_domains', str(num_test_domains),
                '--num_classes', str(num_classes),
                '--resume_iter', str(170000)])


