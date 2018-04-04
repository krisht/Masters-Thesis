def get_triplets(size=10):
     A = []
     P = []
     N = []

     for _ in range(size):
          choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
          neg_choices = list(choices)
          choice = random.choice(choices)
          neg_choices.remove(choice)

          if choice == 'bckg':
               a = np.load(random.choice(bckg))
               p = np.load(random.choice(bckg))
          elif choice == 'eybl':
               a = np.load(random.choice(eybl))
               p = np.load(random.choice(eybl))
          elif choice == 'gped':
               a = np.load(random.choice(gped))
               p = np.load(random.choice(gped))
          elif choice == 'spsw':
               a = np.load(random.choice(spsw))
               p = np.load(random.choice(spsw))
          elif choice == 'pled':
               a = np.load(random.choice(pled))
               p = np.load(random.choice(pled))
          else:
               a = np.load(random.choice(artf))
               p = np.load(random.choice(artf))

          neg_choice = random.choice(neg_choices)

          if neg_choice == 'bckg':
               n = np.load(random.choice(bckg))
          elif neg_choice == 'eybl':
               n = np.load(random.choice(eybl))
          elif neg_choice == 'gped':
               n = np.load(random.choice(gped))
          elif neg_choice == 'spsw':
               n = np.load(random.choice(spsw))
          elif neg_choice == 'pled':
               n = np.load(random.choice(pled))
          else:
               n = np.load(random.choice(artf))

          key = choice + choice + neg_choice

          if key in count_of_triplets:
               count_of_triplets[key]+=1
          else:
               count_of_triplets[key] = 1

          a = norm_op(a, axisss=0)
          p = norm_op(p, axisss=0)
          n = norm_op(n, axisss=0)
          A.append(a)
          P.append(p)
          N.append(n)


     A = np.asarray(A)
     P = np.asarray(P)
     N = np.asarray(N)
     return A, P, N
