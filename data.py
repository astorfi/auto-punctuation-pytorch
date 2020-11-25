import os

def apply_punc(text_input, punctuation):
    assert len(text_input) == len(punctuation), "input string has differnt length from punctuation list" + "".join(
        text_input) + str(punctuation) + str(len(text_input)) + ";" + str(len(punctuation))
    result = ""
    for char1, char2 in zip(text_input, punctuation):
        if char2 == "<cap>":
            result += char1.upper()
        elif char2 == "<nop>":
            result += char1
        else:
            result += char2 + char1
    return result


def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    input_length = len(string_input)
    i = 0
    while i < input_length:
        char = string_input[i]
        if char.isupper():
            output_source.append("<cap>")
            input_source.append(char.lower())

        if char in output_chars:
            output_source.append(char)
            if i < input_length - 1:
                input_source.append(string_input[i + 1])
            else:
                input_source.append(" ")
            i += 1

        if not char.isupper() and char not in output_chars and char in input_chars:
            input_source.append(char)
            output_source.append("<nop>")

        i += 1
    return input_source, output_source
