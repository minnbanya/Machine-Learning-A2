from braintree.resource import Resource as Resource

class ConnectedMerchantStatusTransitioned(Resource):
    def __init__(self, gateway, attributes) -> None: ...
    @property
    def merchant_id(self): ...
