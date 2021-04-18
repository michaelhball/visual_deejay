# Visual Deejay

<h3 align="center">
<p>explorations into real-time audiovisual GANs (& how DJs can collaborate with them) </p>
</h3>

:construction: Under construction :construction:

This is my ever-expanding experimentation with building tools for music visualisation. The end result 
will be an end-to-end system for live visualisation of a DJ set, but the intermediate problems 
seem to breed their own cans of worms (though each is a fascinating place for investigation).

## The Pipeline

* [Extract controller features](/visual_deejay/video_feature_extraction.py) (from screen recording of Rekordbox) at each time stamp
* [Extract tracklist](/visual_deejay/tracklist.py)
* For each track in the tracklist:
    * [extract audio features](/visual_deejay/audio_feature_extraction.py)
    * use extracted audio features to feed StyleGAN to generate frame-by-frame visualisation (this 
    will constitute the default representation of a track, & will be later modified in the mixing
    phase) 
* For each time step in the mix, use controller features + default track visualisations 
of all tracks currently playing to create 'mixed' visualisation
 