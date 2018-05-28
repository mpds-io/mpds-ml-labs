
import random
from common import KNN_TABLE
from prediction import periodic_elements, periodic_numbers


def knn_sample(db_handle, prop_ranges_dict):

    query = """
    WITH precise AS (
    SELECT els FROM {table} WHERE
        {z_min} <= z AND z <= {z_max} AND
        {y_min} <= y AND y <= {y_max} AND
        {x_min} <= x AND x <= {x_max} AND
        {k_min} <= k AND k <= {k_max} AND
        {w_min} <= w AND w <= {w_max} AND
        {m_min} <= m AND m <= {m_max} AND
        {d_min} <= d AND d <= {d_max} AND
        {t_min} <= t AND t <= {t_max} LIMIT 1000
    )
    SELECT els FROM precise
    UNION ALL
    SELECT els FROM {table} WHERE (SELECT COUNT(*) FROM precise)=0 AND
        {z_min} - 70 <= z AND z <= {z_max} + 70 AND
        {y_min} - 70 <= y AND y <= {y_max} + 70 AND
        {x_min} - 60 <= x AND x <= {x_max} + 60 AND
        {k_min} - 70 <= k AND k <= {k_max} + 70 AND
        {w_min} - 33 <= w AND w <= {w_max} + 33 AND
        {m_min} - 250 <= m AND m <= {m_max} + 250 AND
        {d_min} - 200 <= d AND d <= {d_max} + 200 AND
        {t_min} - 33 <= t AND t <= {t_max} + 33 LIMIT 1000
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
    result = result[:7]

    return result
