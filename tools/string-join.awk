# if line ends with a " store line in buffer
# if line starts with " and buffer is non empty, join to buffer and remove "[[:space:]]"
# if neither start or end, print buffer if non empty, print line 
# if we see "clang-format off" turn off processing
# if we see "clang-forma on" turn back on processing

function clear_buffer() {
    if (buffer) print buffer
    buffer=""
    in_string=0
}

# don't allow \n" to be replaced, to keep user enforced breaks
function join_strings(buf) {
    #  can't get negative lookbehind to work in awk, so instead
    gsub("\\\\n\"", "REPLACE_HACK_24601", buf)
    gsub("\\\\\"", "REPLACE_HACK_24602", buf)
    gsub("\"[[:space:]]*\"", "", buf)
    gsub("REPLACE_HACK_24601", "\\n\"", buf)
    gsub("REPLACE_HACK_24602", "\\\"", buf)
    buffer=buf
}

BEGIN {
    enabled=1
    in_string=0 
}

# clang-format on/off 
/\/\/ clang-format off/    { enabled=0; clear_buffer(); print $0; next }
/\/\/ clang-format on/     { enabled=1; print $0; next }

{ 
    string_start=0
    string_end=0
    if (!enabled) {
        print $0
        next
    }
}

# line starts with a string quote (ignoring leading whitespace)
/^[[:space:]]*\"/    { string_start=1 }
# line ends with a string quote (ignoring trailing whitespace)
/\"[[:space:]]*$/    { string_end=1 }

# if we're in a string and this line starts with string : join
string_start && in_string {
    buffer=buffer $0
    join_strings(buffer)
    next
}

# if current line ends on a string quote, then start buffer
string_end && !in_string { 
    buffer=$0
    in_string=1
    next
}

# if we were in a string, but not now
in_string && !string_start {
    clear_buffer()
}

    # gsub(/[aeiou]/, "%")

in_string==0 {
    print $0
}

END {
    if (buffer) print buffer
}
