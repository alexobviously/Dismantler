# Dismantler

I made this in 2013/14 for an undergrad university assignment. It was actually a composition assignment but I somehow ended up writing this. I'm revisiting it in 2023 out of curiosity. The code is basically the same as before but I made it work with Python 3 and modern numpy.

I wrote this out basically because I wanted to remember what this thing did, and now I do. I'm glad it still works and does something relatively interesting. It is still just a script that doesn't take any parameters or anything, so if you want to run it for some reason, go and edit the globals in `main.py` (lol). I'm unlikely to ever expand this, I just wanted it to be able to run again. There is also [iota](https://github.com/alexobviously/iota), which is a related and slightly more developed project, which I might actually improve at some point.

* Originally this was vaguely inspired by chance/indeterminacy based music like John Cage etc.
* That guy and his mates had some pieces where they took a bunch of samples (originally on tape) and arranged them according to some chance procedure.
* I wanted to do that sort of thing but instead taking all of the samples from a single cohesive audio file (e.g. a song), and grouping them somehow.
* This script groups the samples with unsupervised machine learning, specifically k-means clustering, where the features used are randomly selected bins from a spectrogram of the samples.
* A number of separate tracks are generated, each composed of samples all coming from the same group as determined by clustering.
* The density of samples throughout each track varies periodically based on a sine wave at a random frequency.
* Some other stuff is randomised.
* The tracks are then mixed together.

There is also a version that uses MFCCs as features, but I haven't updated that one. I don't really remember why I made it separate, or at all. I think it might not have worked as well.