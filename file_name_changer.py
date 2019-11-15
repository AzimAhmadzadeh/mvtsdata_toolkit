import os
import utils.meta_data_getter as mdg

# This script is written to change the mvts's file names to the form
# that we agreed on. An example of the expected naming convention
# would be:
# lab[B]1.0@1053_id[345]_st[2011-01-24T03:24:00]_et[2011-01-24T11:12:00].csv
# that was changed from:
# B1.0@1053_ar345_s2011-01-24T03:24:00_e2011-01-24T11:12:00.csv
#
# @TODO: This script is NOT a part of the toolkit and MUST be removed
#   as soon as we are sure that we resolved the naming issue.


def main():
    root = './pet_datasets/subset_partition3'
    for filename in os.listdir(root):
        res = filename.replace('_ar', '_id[')
        res = res.replace('_s', ']_st[')
        res = res.replace('_e', ']_et[')
        res = res.replace('.csv', '].csv')
        if res.startswith('N'):
            res = 'lab[{}]{}'.format(res[0:2], res[2:])
        else:
            res = 'lab[{}]{}'.format(res[0], res[1:])

        old_path = os.path.join(root, filename)
        new_path = os.path.join(root, res)
        os.rename(old_path, new_path)


if __name__ == "__main__":
    main()
