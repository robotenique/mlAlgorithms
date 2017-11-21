import re

from porterStemmer import porterStemmer
from getVocabList import getVocabList



def processEmail(email_contents):
    """preprocesses the body of an email and returns a list of word_indices
    """

    # Load Vocabulary
    vocabList = getVocabList()
    vocabDict = {w: idx for idx, w in enumerate(vocabList)}
    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([chr(10) chr(10)]))
    # email_contents = email_contents(hdrstart(1):end)

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    rx = re.compile('<[^<>]+>|\n')
    email_contents = rx.sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    rx = re.compile('[0-9]+')
    email_contents = rx.sub('number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    rx = re.compile('(http|https)://[^\s]*')
    email_contents = rx.sub('httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    rx = re.compile('[^\s]+@[^\s]+')
    email_contents = rx.sub('emailaddr ', email_contents)

    # Handle $ sign
    rx = re.compile('[$]+')
    email_contents = rx.sub('dollar ', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('==== Processed Email ====\n')

    # Process file
    l = 0

    # Remove any non alphanumeric characters
    rx = re.compile('[^a-zA-Z0-9 ]')
    email_contents = rx.sub('', email_contents).split()

    for str_token in email_contents:

        # Tokenize and also get rid of any punctuation
        # str = re.split('[' + re.escape(' @$/#.-:&*+=[]?!(){},''">_<#')
        #                                + chr(10) + chr(13) + ']', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            str_token = porterStemmer(str_token.strip())
        except:
            str_token = ''
            continue

        # Skip the word if it is too short
        if len(str_token) < 1:
            continue

        # Get index
        idx = vocabDict.get(str_token, -1)
        # Skip the word if not in the dict
        if idx == -1:
            continue

        word_indices.append(idx)

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(str_token) + 1) > 78:
            print(str_token)
            l = 0
        else:
            print(str_token)
            l = l + len(str_token) + 1

    # Print footer
    print('\n=========================')
    return word_indices
