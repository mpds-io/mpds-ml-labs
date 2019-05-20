
from __future__ import division
import random
from copy import deepcopy

from common import KNN_TABLE
from prediction import periodic_elements, periodic_numbers
from prediction_ranges import prediction_margins


def knn_sample(db_handle, user_ranges_dict):

    prop_ranges_dict = deepcopy(user_ranges_dict)

    for prop_id in ['x', 'w', 't']:
        # NB. internally treated as *10 to fit SMALLINT
        prop_ranges_dict[prop_id + '_min'] *= 10
        prop_ranges_dict[prop_id + '_max'] *= 10

    # NB. internally treated as *100 to fit SMALLINT
    prop_ranges_dict['i_min'] *= 100
    prop_ranges_dict['i_max'] *= 100

    query = """
    WITH precise AS (
    SELECT els FROM {table} WHERE
        {z_min}::SMALLINT <= z AND z <= {z_max}::SMALLINT AND
        {y_min}::SMALLINT <= y AND y <= {y_max}::SMALLINT AND
        {x_min}::SMALLINT <= x AND x <= {x_max}::SMALLINT AND
        {k_min}::SMALLINT <= k AND k <= {k_max}::SMALLINT AND
        {w_min}::SMALLINT <= w AND w <= {w_max}::SMALLINT AND
        {m_min}::SMALLINT <= m AND m <= {m_max}::SMALLINT AND
        {d_min}::SMALLINT <= d AND d <= {d_max}::SMALLINT AND
        {t_min}::SMALLINT <= t AND t <= {t_max}::SMALLINT AND
        {i_min}::SMALLINT <= i AND i <= {i_max}::SMALLINT AND
        {o_min}::SMALLINT <= o AND o <= {o_max}::SMALLINT
        LIMIT 3000
    )
    SELECT els FROM precise
    UNION ALL
    SELECT els FROM {table} WHERE (SELECT COUNT(*) FROM precise)=0 AND
        ({z_min} - {z_margin})::SMALLINT <= z AND z <= ({z_max} + {z_margin})::SMALLINT AND
        ({y_min} - {y_margin})::SMALLINT <= y AND y <= ({y_max} + {y_margin})::SMALLINT AND
        ({x_min} - {x_margin})::SMALLINT <= x AND x <= ({x_max} + {x_margin})::SMALLINT AND
        ({k_min} - {k_margin})::SMALLINT <= k AND k <= ({k_max} + {k_margin})::SMALLINT AND
        ({w_min} - {w_margin})::SMALLINT <= w AND w <= ({w_max} + {w_margin})::SMALLINT AND
        ({m_min} - {m_margin})::SMALLINT <= m AND m <= ({m_max} + {m_margin})::SMALLINT AND
        ({d_min} - {d_margin})::SMALLINT <= d AND d <= ({d_max} + {d_margin})::SMALLINT AND
        ({t_min} - {t_margin})::SMALLINT <= t AND t <= ({t_max} + {t_margin})::SMALLINT AND
        ({i_min} - {i_margin})::SMALLINT <= i AND i <= ({i_max} + {i_margin})::SMALLINT AND
        ({o_min} - {o_margin})::SMALLINT <= o AND o <= ({o_max} + {o_margin})::SMALLINT
        LIMIT 3000
    """.format(
        table=KNN_TABLE,
        z_margin=prediction_margins['z'],
        y_margin=prediction_margins['y'],
        x_margin=prediction_margins['x'],
        k_margin=prediction_margins['k'],
        w_margin=prediction_margins['w'],
        m_margin=prediction_margins['m'],
        d_margin=prediction_margins['d'],
        t_margin=prediction_margins['t'],
        i_margin=prediction_margins['i'],
        o_margin=prediction_margins['o'],
        **prop_ranges_dict
    )
    #print(query)
    db_handle.execute(query)

    result = []
    for deck in db_handle.fetchall():
        els = [periodic_elements[periodic_numbers.index(int(pn))] for pn in deck[0].split(',') if int(pn) != 0]
        result.append(els)

    #print("KNN LENGTH: %s" % len(result))

    random.shuffle(result)
    return result


if __name__ == "__main__":

    # test knn sampling by generating the random ranges
    # of properties and requesting the elements within (or nearby) them

    from pprint import pprint
    import time
    from common import connect_database
    from prediction_ranges import prediction_ranges

    cursor, connection = connect_database()

    sample = {}

    for prop_id in prediction_ranges:
        dice = random.choice([0, 1])
        bound = (prediction_ranges[prop_id][1] - prediction_ranges[prop_id][0]) / 3
        if dice:
            sample[prop_id + '_min'] = prediction_ranges[prop_id][0] + bound * 2
            sample[prop_id + '_max'] = prediction_ranges[prop_id][1]
        else:
            sample[prop_id + '_min'] = prediction_ranges[prop_id][0]
            sample[prop_id + '_max'] = prediction_ranges[prop_id][0] + bound

    for prop_id in ['x', 'w', 't']:
        # NB. internally treated as *10 to fit SMALLINT
        sample[prop_id + '_min'] *= 10
        sample[prop_id + '_max'] *= 10

    # NB. internally treated as *100 to fit SMALLINT
    sample['i_min'] *= 100
    sample['i_max'] *= 100

    #for prop_id in ['z', 'y', 'x', 'k', 'w', 'm', 'd', 't', 'i', 'o']:
    #    print("%s E ( %s --- %s )" % (prop_id, sample[prop_id + '_min'], sample[prop_id + '_max']))

    # duplicate query below as it provides more debug info (precise vs. approx.)
    query = """
    WITH precise AS (
    SELECT els FROM {table} WHERE
        {z_min}::SMALLINT <= z AND z <= {z_max}::SMALLINT AND
        {y_min}::SMALLINT <= y AND y <= {y_max}::SMALLINT AND
        {x_min}::SMALLINT <= x AND x <= {x_max}::SMALLINT AND
        {k_min}::SMALLINT <= k AND k <= {k_max}::SMALLINT AND
        {w_min}::SMALLINT <= w AND w <= {w_max}::SMALLINT AND
        {m_min}::SMALLINT <= m AND m <= {m_max}::SMALLINT AND
        {d_min}::SMALLINT <= d AND d <= {d_max}::SMALLINT AND
        {t_min}::SMALLINT <= t AND t <= {t_max}::SMALLINT AND
        {i_min}::SMALLINT <= i AND i <= {i_max}::SMALLINT AND
        {o_min}::SMALLINT <= o AND o <= {o_max}::SMALLINT
        LIMIT 3000
    )
    SELECT 1, els FROM precise
    UNION ALL
    SELECT 0, els FROM {table} WHERE (SELECT COUNT(*) FROM precise)=0 AND
        ({z_min} - {z_margin})::SMALLINT <= z AND z <= ({z_max} + {z_margin})::SMALLINT AND
        ({y_min} - {y_margin})::SMALLINT <= y AND y <= ({y_max} + {y_margin})::SMALLINT AND
        ({x_min} - {x_margin})::SMALLINT <= x AND x <= ({x_max} + {x_margin})::SMALLINT AND
        ({k_min} - {k_margin})::SMALLINT <= k AND k <= ({k_max} + {k_margin})::SMALLINT AND
        ({w_min} - {w_margin})::SMALLINT <= w AND w <= ({w_max} + {w_margin})::SMALLINT AND
        ({m_min} - {m_margin})::SMALLINT <= m AND m <= ({m_max} + {m_margin})::SMALLINT AND
        ({d_min} - {d_margin})::SMALLINT <= d AND d <= ({d_max} + {d_margin})::SMALLINT AND
        ({t_min} - {t_margin})::SMALLINT <= t AND t <= ({t_max} + {t_margin})::SMALLINT AND
        ({i_min} - {i_margin})::SMALLINT <= i AND i <= ({i_max} + {i_margin})::SMALLINT AND
        ({o_min} - {o_margin})::SMALLINT <= o AND o <= ({o_max} + {o_margin})::SMALLINT
        LIMIT 3000
    """.format(
        table=KNN_TABLE,
        z_margin=prediction_margins['z'],
        y_margin=prediction_margins['y'],
        x_margin=prediction_margins['x'],
        k_margin=prediction_margins['k'],
        w_margin=prediction_margins['w'],
        m_margin=prediction_margins['m'],
        d_margin=prediction_margins['d'],
        t_margin=prediction_margins['t'],
        i_margin=prediction_margins['i'],
        o_margin=prediction_margins['o'],
        **sample
    )

    start_time = time.time()
    cursor.execute(query)
    #print(query)
    print("Query done in %1.2f sc" % (time.time() - start_time))

    result = []
    precise = 0

    for deck in cursor.fetchall():
        precise = deck[0]
        els = [periodic_elements[periodic_numbers.index(int(pn))] for pn in deck[1].split(',') if int(pn) != 0]
        result.append(els)

    print("PRECISE" if precise else "APPROX.")
    print("Total:", len(result))

    random.shuffle(result)
    pprint(result[:5])

    connection.close()
