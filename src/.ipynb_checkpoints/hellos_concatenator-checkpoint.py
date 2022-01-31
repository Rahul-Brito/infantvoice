#seedno=1
#seed(seedno)
#samples= []
#for filename in os.listdir(directory):
#    if filename.endswith(".wav"):
#        samples.append(filename[0:4])

#samples = np.reshape(samples, (17,30))
#[shuffle(f) for f in samples[:]]
#samples = np.reshape(samples, (17,10,3))

#for f in samples[:][:]:
#    for i in f:
#        clear_output(wait=True)
#        first = AudioSegment.from_wav(os.path.join(directory, i[0]+'.wav'))
#        second = AudioSegment.from_wav(os.path.join(directory, i[1]+'.wav'))
#        third = AudioSegment.from_wav(os.path.join(directory, i[2]+'.wav'))
#       concat = first + second + third
#        concat.export(os.path.join(dataloc,i[0]+'_'+i[1] + '_' + i[2]+'.wav'), format="wav")
#        print("Processing" + i[0]+'_'+i[1] + '_' + i[2])