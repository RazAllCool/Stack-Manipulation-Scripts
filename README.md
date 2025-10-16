>TrackGradientExtract: this doesn't work yet, but I felt like temporal color coding could be used
to track larvae in a single image, by extracting color gradients and clustering them
into trajectories.
>
>TiffToMulti: Turns a directory of single tiff frames into a multistack
>
>RegisterAVideo: Fixes shakycam-- isn't nearly as good as just using MutliStackRegistration or other on FIJI, but I didn't know that was available and all I had was a hammer
>
>DenoiseTiff: Much like my registration script, I was unaware FIJI was going to solve all my problems, and I knew how to learn python. This one uses total variation denoising, which does a good job of preserving edges IMO
