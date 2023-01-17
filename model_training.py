import pandas as pd
from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE

if __name__ == '__main__':
    # Set the parameters that will be used later
    input_folder = './'
    output_folder = './'
    input_file = 'Power_grid_real_data.csv'
    gan = 'CopulaGAN'  # either 'CopulaGAN', 'CTGAN', 'TVAE', or 'GaussianCopula'
    # Each of natural class and attack class will be generating this many of data samples
    # Note the two classes will be generated and stored in different csv files
    num_rows = 100000
    # params for gan model
    epoch = 500
    batch_size = 500
    embedding_dim = 256
    generator_dim = (512, 512)
    discriminator_dim = (512, 512)
    generator_lr = 0.0003
    discriminator_lr = 0.0003
    discriminator_steps = 5

    data = pd.read_csv(input_folder + input_file)
    print("training the model")
    if (gan == 'CopulaGAN'):
        model = CopulaGAN(epochs=epoch,
                          batch_size=batch_size,
                          embedding_dim=embedding_dim,
                          generator_dim=generator_dim,
                          discriminator_dim=discriminator_dim,
                          generator_lr=generator_lr,
                          discriminator_lr=discriminator_lr,
                          discriminator_steps=discriminator_steps,
                          verbose=True)
    elif (gan == 'CTGAN'):
        model = CTGAN(epochs=epoch,
                      batch_size=batch_size,
                      embedding_dim=embedding_dim,
                      generator_dim=generator_dim,
                      discriminator_dim=discriminator_dim,
                      generator_lr=generator_lr,
                      discriminator_lr=discriminator_lr,
                      discriminator_steps=discriminator_steps,
                      verbose=True)
    elif (gan == 'GaussianCopula'):
        model = GaussianCopula(epochs=epoch,
                               batch_size=batch_size,
                               embedding_dim=embedding_dim,
                               generator_dim=generator_dim,
                               discriminator_dim=discriminator_dim,
                               generator_lr=generator_lr,
                               discriminator_lr=discriminator_lr,
                               discriminator_steps=discriminator_steps,
                               verbose=True)
    elif (gan == 'TVAE'):
        model = TVAE(epochs=epoch,
                     batch_size=batch_size,
                     embedding_dim=embedding_dim,
                     generator_dim=generator_dim,
                     discriminator_dim=discriminator_dim,
                     generator_lr=generator_lr,
                     discriminator_lr=discriminator_lr,
                     discriminator_steps=discriminator_steps,
                     verbose=True)
    model.fit(data)
    print("saving the model")
    model.save(output_folder + 'name_of_gan' + '.pkl')

    print("loading the model")
    if (gan == 'CopulaGAN'):
        loaded = CopulaGAN.load(output_folder + 'name_of_gan' + '.pkl')
    elif (gan == 'CTGAN'):
        loaded = CTGAN.load(output_folder + 'name_of_gan' + '.pkl')
    elif (gan == 'GaussianCopula'):
        loaded = GaussianCopula.load(output_folder + 'name_of_gan' + '.pkl')
    elif (gan == 'TVAE'):
        loaded = TVAE.load(output_folder + 'name_of_gan' + '.pkl')

    print("generating data")
    conditions = Condition({'marker': 'Attack'}, num_rows=num_rows)
    new_data_attack = loaded.sample_conditions(conditions=[conditions])

    conditions = Condition({'marker': 'Natural'}, num_rows=num_rows)
    new_data_natural = loaded.sample_conditions(conditions=[conditions])

    print("saving the dataset")
    new_data_attack.to_csv(output_folder + '_attack_' + str(num_rows) + '.csv')
    new_data_natural.to_csv(output_folder + '_natural_' + str(num_rows) + '.csv')
