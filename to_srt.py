import datetime


def to_srt(filename):
    file = open(filename)
    lines = file.readlines()
    i = 0
    for line in lines:
        with open('output.srt', 'a+') as f:
            f.write(str(i) + '\n')
            time_stamp = datetime.timedelta(milliseconds=200 + i * 10).__str__()[:11] + ' --> ' + \
                         datetime.timedelta(milliseconds=200 + i * 10 + 10) .__str__()[:11] + '\n'
            time_stamp = time_stamp.replace('.', ',')
            f.write(time_stamp)
            if line == '1 0\n':
                f.write('NO SPEECH\n\n')
            # elif line == '0 1 0\n':
            #     f.write('NO SPEECH\n\n')
            # if line == '0 0 1\n':
            else:
                f.write('SPEECH\n\n')

            i += 1
            print(i)


to_srt('first_results.txt')
