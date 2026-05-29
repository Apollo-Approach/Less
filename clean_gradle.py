import os
f='gradle.properties'
if os.path.exists(f):
    lines = open(f, 'r').readlines()
    open(f, 'w').write(''.join([l for l in lines if 'github_token' not in l]))
