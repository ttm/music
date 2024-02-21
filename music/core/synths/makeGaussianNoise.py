import numpy as n

def makeGaussianNoise(self,mean,std,DUR=2):
    Lambda = DUR*self.samples_beat # Lambda sempre par
    df = self.samplerate/float(Lambda)
    MEAN=mean
    STD=.1
    coefs = n.exp(1j*n.random.uniform(0, 2*n.pi, Lambda))
    # real par, imaginaria impar
    coefs[Lambda/2+1:] = n.real(coefs[1:Lambda/2])[::-1] - 1j * \
        n.imag(coefs[1:Lambda/2])[::-1]
    coefs[0] = 0.  # sem bias
    if Lambda%2==0:
        coefs[Lambda/2] = 0.  # freq max eh real simplesmente

    # as frequências relativas a cada coeficiente
    # acima de Lambda/2 nao vale
    fi = n.arange(coefs.shape[0])*df
    f0 = 15.  # iniciamos o ruido em 15 Hz
    f1=(mean-std/2)*3000
    f2=(mean+std/2)*3000
    i1 = n.floor(f1/df)  # primeiro coef a valer
    i2 = n.floor(f2/df)  # ultimo coef a valer
    coefs[:i1] = n.zeros(i1)
    coefs[i2:]=n.zeros(len(coefs[i2:]))

    # obtenção do ruído em suas amostras temporais
    ruido = n.fft.ifft(coefs)
    r = n.real(ruido)
    r = ((r-r.min())/(r.max()-r.min()))*2-1

    # fazer tre_freq variar conforme measures2
    return r
