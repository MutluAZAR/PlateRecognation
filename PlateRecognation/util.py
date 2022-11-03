import datetime
import math
import numpy as np
from configs import *

enum = {
    "m": tuple,
    "id": int,
    "usage": bool
}

Ids = []
Temp_Records = []
Case = None
DetectedZones = []


def choose_records(records):
    returned_value = []
    for i in records:
        if i[4] > CONFIDENCE_THRESHOLD:
            returned_value.append(list(i))
    return returned_value


def create_ids(records):
    if len(Ids) > 0:
        for m in Ids:
            m['usage'] = False

    def calc_centers():
        data = []
        for i in records:
            c = ((i[0] + i[2]) / 2, (i[1] + i[3]) / 2)
            data.append(c)

        return data

    def calc_indexes():
        distances = []
        for i in centers:
            row = []
            for xk in Ids:
                dist = math.sqrt((i[0] - xk['m'][0]) ** 2 + (i[1] - xk['m'][1]) ** 2)
                row.append(dist)
            distances.append(row)

        indexs = []
        for f in distances:
            index = np.argmin(np.array(f))
            indexs.append((index, min(f)))

        return indexs

    centers = calc_centers()
    subs = len(records) - len(Ids)

    if subs == 0:
        indexes = calc_indexes()
        for t, (x, k) in enumerate(zip(indexes, records)):
            k.append(centers[t])
            k.append(Ids[x[0]]['id'])
            Ids[x[0]]['m'] = centers[t]
            Ids[x[0]]['usage'] = True

    if subs > 0:
        if len(Ids) == 0:
            for p, k in enumerate(centers):
                ids = int(datetime.datetime.now().strftime('%S%f'))
                e = {'m': k, 'id': ids, 'usage': True}
                Ids.append(e)
                records[p].append(k)
                records[p].append(ids)
        if True:
            indexes = calc_indexes()
            ri = []
            for t, (x, k) in enumerate(zip(indexes, records)):
                ri.append(x[1])
                k.append(centers[t])
                k.append(Ids[x[0]]['id'])
                Ids[x[0]]['m'] = centers[t]
                Ids[x[0]]['usage'] = True

            for _ in range(subs):
                rindex = np.argmax(np.array(ri))
                ids = int(datetime.datetime.now().strftime('%S%f'))
                records[rindex][-1] = ids
                e = {'m': records[rindex][-2], 'id': ids, 'usage': True}
                Ids.append(e)
                ri.pop(rindex)

    if subs < 0:
        indexes = calc_indexes()
        for t, (x, k) in enumerate(zip(indexes, records)):
            k.append(centers[t])
            k.append(Ids[x[0]]['id'])
            Ids[x[0]]['m'] = centers[t]
            Ids[x[0]]['usage'] = True

    for y, x in enumerate(Ids):
        if not x['usage']:
            Ids.pop(y)

    return records
