import math
import numpy as np
import pandas as pd
import cv2  # pip install opencv-python
from clustcr import Clustering  # https://github.com/svalkiers/clusTCR


def label_to_char(x):
    n = int(x[1:])
    return chr(n + 300)


def char_to_label(x):
    s = ''
    for c in x:
        n = ord(c) - 300
        s += ' E' + str(n)
    return s[1:]


def label_to_color(val, max_val=170):
    i = (val * 255 / max_val)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    return r, g, b


def paint_clusters(clusters):
    # create image
    img = np.ones((800, len(clusters) * 300, 3), dtype='uint8') * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(130):
        color = label_to_color(i)
        cv2.rectangle(img, (i * 20 + 50, 20), (i * 20 + 90, 40), color, -1)

    x0 = 50 - 300
    k = 0

    for cluster in clusters:
        x0 += 300
        y0 = 130
        k += 1

        # paint cluster number
        cv2.putText(img, 'cluster ' + str(k), (x0, y0 - 10), font, 1, (0, 0, 0), 2)

        # sort sequences in cluster by length
        cluster.sort(reverse=True, key=lambda x: len(x))

        # paint cluster sequence
        for sequence in cluster:
            y0 += 40
            sequence_labels = [int(n) for n in sequence.replace('E', '').split(' ')]

            for i in range(len(sequence_labels)):
                x = x0 + i * 20
                y = y0
                color = label_to_color(sequence_labels[i])
                cv2.rectangle(img, (x, y), (x + 20, y + 20), color, -1)

    cv2.imwrite('result.jpg', img)


# read data and group labels by id
df = pd.read_excel('sequence of activities.xlsx')
df = df.drop(columns=['TimeValue'])
df['label'] = df['label'].map(lambda x: label_to_char(x))
df_group = df.groupby(' ID').sum()

# create sequences
data = pd.Series(df_group['label'], index=df_group.index)
sequences = pd.DataFrame({'ID': data.index, 'label': data.values})
sequences['label'] = sequences['label'].map(lambda x: char_to_label(x))
sequences.to_excel('sequences.xlsx', index=False)

# cluster sequences
clustering = Clustering()
output = clustering.fit(data)
output_df = output.clusters_df

# put ids
for i in range(len(output_df)):
    indices = df_group.index[df_group['label'] == output_df.iloc[i, 0]].tolist()
    output_df.at[i, 'ids'] = ' , '.join(map(str, indices))

# save result
output_df['junction_aa'] = output_df['junction_aa'].map(lambda x: char_to_label(x))
output_df.to_excel('output.xlsx')

# visualize
paint_clusters(output.cluster_contents())
