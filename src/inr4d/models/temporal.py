"""Role 4: time encoding.

  A. 'concat': t is the 4th input coordinate.
  B. 'latent': one learnable vector per training frame; think about how to get a
     latent for an UNSEEN time (interpolate neighbouring codes?) and log the decision
     in docs/DECISIONS.md.
"""

# 时间编码决定网络如何表示完整的心动周期。
