from gensid.nn.gnn import APPNPConv, S2GRConv


def get_encoder(encoder_name: str):
    def none_encoder(x, *args, **kwargs):
        return x

    encoders = {None: none_encoder, "appnp": APPNPConv, "s2gr": S2GRConv}
    key = encoder_name.lower() if encoder_name is not None else None

    return encoders[key]
