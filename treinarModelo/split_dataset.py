import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar os metadados
data = pd.read_csv('/dataset/all_data/metadata.csv', header=None, delimiter='|')
data.columns = ['filename', 'transcription']

# Dividir os dados em treinamento, validação e teste
train, temp = train_test_split(data, test_size=0.20, random_state=42)  # 80% para treinamento
validation, test = train_test_split(temp, test_size=0.50, random_state=42)  # 10% para validação, 10% para teste

# Salvar os novos metadados
train.to_csv('/dataset/train/metadata.csv', index=False, header=False, sep='|')
validation.to_csv('/dataset/validation/metadata.csv', index=False, header=False, sep='|')
test.to_csv('/dataset/test/metadata.csv', index=False, header=False, sep='|')

# Você também precisará mover os arquivos de áudio correspondentes para suas respectivas pastas
