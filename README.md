# Visual Deejay

<h3 align="center">
<p>:construction: Explorations into real-time audiovisual GANs :construction: </p>
</h3>

This is my ever-expanding experimentation into generative tools for music visualisation. The 
end result will be an end-to-end system for live visualisation of a DJ set, but the intermediate 
steps seem to continually breed their own cans of worms, albeit interesting ones.

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
 