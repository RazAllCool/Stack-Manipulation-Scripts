>TrackGradientExtract: this doesn't work yet, but I felt like temporal color coding could be used
to track larvae in a single image, by extracting color gradients and clustering them
into trajectories.
>
>TiffToMulti: Turns a directory of single tiff frames into a multistack
>
>RegisterAVideo: Fixes shakycam-- isn't nearly as good as just using MutliStackRegistration or other on FIJI, but I didn't know that was available and all I had was a hammer
>
>DenoiseTiff: Much like my registration script, I was unaware FIJI was going to solve all my problems, and I knew how to learn python. This one uses total variation denoising, which does a good job of preserving edges IMO
>
>DecompressanAVI: It decompresses AVIs. I don't remember why I needed that but here it is...
>
>2_3d_projection: I had the idea that if I turned a binary stack into a 3d projection, I could use watershed, 3d CNN, or a maze solving algorithm to track objects elegantly. This is just part of that. Ultimately, I still think FIJI will solve that problem better, unfortunately.
>
