# How to use the reddit bot /u/USNamesBot

## Get data about name(s)

### Syntax

    !name [name]
    !name [name1/name2/name3]

### Examples

    !name marcus
    !name natalie
    !name louis/lewis
    !name ashley/ashton/ashling

## Search for names based on their features/characteristics

### Syntax

    !search [conditions]

### Examples

You can mix-and-match the below conditions:

#### Regex pattern (Python)

Must be placed last in a series of conditions.

    !search pattern:^a[iyeo]+dh?[eiyoa]n$
    !search gender:f pattern:^al[ei](x|j|ss?|ks|k?z)an

#### Start & end - single string or comma-separated strings

    !search start:ash
    !search start:ma,na
    !search end:der
    !search end:z,s
    !search start:g end:y

#### Contains (uses AND condition) - single string or comma-separated strings

    !search contains:dre
    !search contains:ie,n
    !search start:f contains:r
    !search end:a,ah gender:m

#### Contains-any (uses OR condition) - single string or comma-separated strings

    !search contains-any:eli,elli

### NOT start/end/contains - single string or comma-separated strings (uses AND condition)

    !search ~start:r,s,t
    !search ~end:n,m
    !search ~contains:a,e,i,o,u

#### Order - comma-separated strings

    !search order:j,d,n gender:x
    !search gender:f order:d,l,n

#### Length - hyphen-separated integers

    !search length:3-5 ~contains:s,z,x

#### Gender

Filter on the aggregate gender lean of the name. Options are `m` (masculine), `f` (feminine), and `x` (gender-neutral or unisex).

    !search length:4-7 gender:x start:q
    !search gender:m,x contains:rose

#### Years

Range (inclusive)

    !search gender:m start:k years:1970-1990

Single year

    !search gender:m start:k year:1955

After/before years (inclusive)

    !search gender:m start:k after:1990
    !search gender:m start:k before:1976
