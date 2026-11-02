"""Build the job list, in priority order. Seeds are unique per job."""
import json, sys
sel = json.load(open('results/selected_minima.json'))
A = [m['name'] for m in sel['A']]
B = [m['name'] for m in sel['B']]
sub_A = [A[i] for i in (0, 2, 4, 7, 9, 11)]   # spans the lmax range, for the expensive grids
sub_B = [B[i] for i in (0, 2, 5)]
T = 150000
ADAM_LRS_A = [float(x) for x in sys.argv[1].split(',')]
ADAM_LRS_B = [float(x) for x in sys.argv[2].split(',')]
jobs, seed = [], 5000


def add(exp, land, m, opt, lr, b, R, **kw):
    global seed
    seed += 1
    jobs.append(dict(id=f'{exp}__{m}__{opt}_lr{lr:g}_B{b}' + ''.join(f'_{k}{v}' for k, v in kw.items()),
                     exp=exp, land=land, min=m, opt=opt, lr=lr, B=b, seed=seed, R=R, t_max=T, **kw))


# Base condition eta=0.1, B=16; eta=0.2 is the hot condition where escape competes with in-well relaxation.
for m in A:
    add('E1', 'A', m, 'sgd', 0.1, 16, 256)
for m in B:
    add('E1', 'B', m, 'sgd', 0.1, 16, 256)
for m in A:
    add('E1', 'A', m, 'sgd', 0.2, 16, 128)
for m in B:
    add('E1', 'B', m, 'sgd', 0.2, 16, 128)
for land, ms in (('A', sub_A), ('B', sub_B)):
    for b in (32, 8, 64, 4):
        for m in ms:
            add('E3', land, m, 'sgd', 0.1, b, 128)
    for lr, b in ((0.05, 8), (0.2, 32)):
        for m in ms:
            add('E3', land, m, 'sgd', lr, b, 128)
for lr in ADAM_LRS_A:
    for m in A:
        add('E2', 'A', m, 'adam', lr, 16, 128)
for lr in ADAM_LRS_B:
    for m in B:
        add('E2', 'B', m, 'adam', lr, 16, 128)
for m in sub_A:
    add('ABL', 'A', m, 'sgdm', 0.01, 16, 128)
    add('ABL', 'A', m, 'rmsprop', ADAM_LRS_A[0], 16, 128)
    add('ABL', 'A', m, 'adam', ADAM_LRS_A[0], 16, 128, warm=1)
    add('ABL', 'A', m, 'adam', 0.001, 16, 64)
for m in (A[0], A[4], A[7], A[11]):
    add('ROB', 'A', m, 'sgd', 0.1, 16, 128, qmult=3)
json.dump(jobs, open('results/jobs.json', 'w'), indent=0)
print(len(jobs), 'jobs; unique ids', len({j['id'] for j in jobs}))
