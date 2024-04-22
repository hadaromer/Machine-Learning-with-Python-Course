import json
from utils.utils import *

ALPHABET_SIZE = 26


# Function to check if all words in a phrase are present in the lexicon
def is_words_in_lexicon(phrase, lexicon):
    words = phrase.split()  # Split the phrase into words
    for word in words:
        if word not in lexicon:  # Check if each word is in the lexicon
            return False
    return True


# Function to shift characters in a phrase by a specified number of steps
def shift_characters(phrase, k, alphabet):
    shifted_word = ''

    for char in phrase:
        if char.isalpha():  # Check if the character is alphabetic
            index = (alphabet.index(char) + k) % ALPHABET_SIZE  # Calculate the new index after shifting
            shifted_word += alphabet[index]  # Append the shifted character to the result
        else:
            shifted_word += char  # Keep non-alphabetic characters unchanged

    return shifted_word


# Function to decipher a phrase using a given lexicon and alphabet
def decipher_phrase(phrase, lexicon_filename, abc_filename):
    print(f'starting deciphering using {lexicon_filename} and {abc_filename}')

    # Load lexicon
    check_file_exists(lexicon_filename)  # Check if the lexicon file exists
    with open(lexicon_filename, 'r', encoding='utf8') as lexicon_file:
        lexicon = lexicon_file.read().splitlines()  # Read lexicon file and split by lines

    # Load alphabet
    check_file_exists(abc_filename)  # Check if the alphabet file exists
    with open(abc_filename, 'r', encoding='utf8') as abc_file:
        alphabet = abc_file.read().splitlines()  # Read alphabet file and split by lines

    # Check if the phrase is empty
    if not phrase.strip():
        return {"status": 2}  # Empty phrase status

    deciphered_phrase = ''
    k = None
    for i in range(ALPHABET_SIZE):
        shifted_phrase = shift_characters(phrase, i, alphabet)  # Shift characters in the phrase
        if is_words_in_lexicon(shifted_phrase, lexicon):  # Check if shifted phrase words are in the lexicon
            k = ALPHABET_SIZE - i
            deciphered_phrase = shifted_phrase
            break

    if k is None:  # If K couldn't be determined
        return {"status": 1}  # Error status

    return {"status": 0, "orig_phrase": deciphered_phrase, "K": k}  # Success status with deciphered phrase and K


if __name__ == '__main__':
    with open('config-decipher.json', 'r') as json_file:
        config = json.load(json_file)

    result = decipher_phrase(config['secret_phrase'], config['lexicon_filename'], config['abc_filename'])

    assert result["status"] in {0, 1, 2}

    if result["status"] == 0:
        print(f'deciphered phrase: {result["orig_phrase"]}, K: {result["K"]}')
    elif result["status"] == 1:
        print("cannot decipher the phrase!")
    else:  # result["status"] == 2:
        print("empty phrase")
