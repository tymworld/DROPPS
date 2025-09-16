def validate_extension(filename, extension):

    expanded_extension = extension if extension.startswith('.') else ('.' + extension)
    treated_name = filename if filename.endswith(expanded_extension) else (filename + expanded_extension)
    return treated_name

