import json
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os

# Carregar as configurações do JSON
with open('my_config.json', 'r') as file:
    config = json.load(file)

# Caminhos baseados no arquivo de configuração
data_path = config['datasets'][0]['path'] + config['datasets'][0]['meta_file_train']
train_path = config['train_path'] + '/wavs'
validation_path = config['validation_path'] + '/wavs'
test_path = config['test_path'] + '/wavs'

# Carregar os metadados
data = pd.read_csv(data_path, header=None, delimiter='|')
data.columns = ['filename', 'transcription']

# Dividir os dados em treinamento, validação e teste
train, temp = train_test_split(data, test_size=0.20, random_state=42)
validation, test = train_test_split(temp, test_size=0.50, random_state=42)

# Salvar os novos metadados
train.to_csv(config['train_path'] + '/metadata.csv', index=False, header=False, sep='|')
validation.to_csv(config['validation_path'] + '/metadata.csv', index=False, header=False, sep='|')
test.to_csv(config['test_path'] + '/metadata.csv', index=False, header=False, sep='|')

# Função para mover arquivos
def move_files(df, source_folder, target_folder):
    for filename in df['filename']:
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.move(source_path, target_path)

# Mover arquivos de áudio para as pastas correspondentes
move_files(train, config['dataset_path'] + '/wavs', train_path)
move_files(validation, config['dataset_path'] + '/wavs', validation_path)
move_files(test, config['dataset_path'] + '/wavs', test_path)
