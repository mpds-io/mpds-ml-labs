
import random
from copy import deepcopy

from common import KNN_TABLE
from prediction import periodic_elements, periodic_numbers


N_SAMPLES = 10 # the more samples, the more probability to succeed,
               # but the more time-consuming the scoring

def knn_sample(db_handle, user_ranges_dict):

    prop_ranges_dict = deepcopy(user_ranges_dict)

    for prop_id in ['x', 'w', 't']:
        # NB. internally treated as *10 to fit SMALLINT
        prop_ranges_dict[prop_id + '_min'] *= 10
        prop_ranges_dict[prop_id + '_max'] *= 10

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
        {t_min}::SMALLINT <= t AND t <= {t_max}::SMALLINT LIMIT 3000
    )
    SELECT els FROM precise
    UNION ALL
    SELECT els FROM {table} WHERE (SELECT COUNT(*) FROM precise)=0 AND
        ({z_min} - 70)::SMALLINT <= z AND z <= ({z_max} + 70)::SMALLINT AND
        ({y_min} - 70)::SMALLINT <= y AND y <= ({y_max} + 70)::SMALLINT AND
        ({x_min} - 60)::SMALLINT <= x AND x <= ({x_max} + 60)::SMALLINT AND
        ({k_min} - 70)::SMALLINT <= k AND k <= ({k_max} + 70)::SMALLINT AND
        ({w_min} - 33)::SMALLINT <= w AND w <= ({w_max} + 33)::SMALLINT AND
        ({m_min} - 250)::SMALLINT <= m AND m <= ({m_max} + 250)::SMALLINT AND
        ({d_min} - 200)::SMALLINT <= d AND d <= ({d_max} + 200)::SMALLINT AND
        ({t_min} - 33)::SMALLINT <= t AND t <= ({t_max} + 33)::SMALLINT LIMIT 3000
    """.format(
        table=KNN_TABLE, **prop_ranges_dict
    )
    #print query
    db_handle.execute(query)

    result = []
    for deck in db_handle.fetchall():
        els = [periodic_elements[periodic_numbers.index(int(pn))] for pn in deck[0].split(',') if int(pn) != 0]
        result.append(els)

    #print "KNN LENGTH: %s" % len(result)

    random.shuffle(result)
    result = result[:N_SAMPLES]

    return result
