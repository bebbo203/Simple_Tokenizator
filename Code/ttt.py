def _get_bigrams(sentence_list):
      bigrams = {}
      # for each sentence
      for s in sentence_list:
        # for each bigram
        for k in range(len(s)-1):
          bigrams[s[k:k+2]] = 1.0
      return bigrams.keys()


print(_get_bigrams(["Ciao a tutti amici!", "io sono Roberto e vivo a Frosinone"]))
