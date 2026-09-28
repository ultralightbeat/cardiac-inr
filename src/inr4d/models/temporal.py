"""Role 4: time encoding.

  A. 'concat': t is the 4th input coordinate.
  B. 'latent': one learnable vector per training frame; think about how to get a
     latent for an UNSEEN time (interpolate neighbouring codes?) and log the decision
     in docs/DECISIONS.md.
"""
