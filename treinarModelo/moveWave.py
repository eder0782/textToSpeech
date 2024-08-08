import shutil
import os

def move_files(df, source_folder, target_folder):
    for filename in df['filename']:
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.move(source_path, target_path)

# Mover arquivos de áudio para as pastas correspondentes
move_files(train, '/dataset/all_data/wavs', '/dataset/train/wavs')
move_files(validation, '/dataset/all_data/wavs', '/dataset/validation/wavs')
move_files(test, '/dataset/all_data/wavs', '/dataset/test/wavs')
