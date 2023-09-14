class ErrorCodes:
    class Address:
        CannotBeBlank: str
        CompanyIsInvalid: str
        CompanyIsTooLong: str
        CountryCodeAlpha2IsNotAccepted: str
        CountryCodeAlpha3IsNotAccepted: str
        CountryCodeNumericIsNotAccepted: str
        CountryNameIsNotAccepted: str
        ExtendedAddressIsInvalid: str
        ExtendedAddressIsTooLong: str
        FirstNameIsInvalid: str
        FirstNameIsTooLong: str
        InconsistentCountry: str
        IsInvalid: str
        LastNameIsInvalid: str
        LastNameIsTooLong: str
        LocalityIsInvalid: str
        LocalityIsTooLong: str
        PostalCodeInvalidCharacters: str
        PostalCodeIsInvalid: str
        PostalCodeIsRequired: str
        PostalCodeIsRequiredForCardBrandAndProcessor: str
        PostalCodeIsTooLong: str
        RegionIsInvalid: str
        RegionIsTooLong: str
        StateIsInvalidForSellerProtection: str
        StreetAddressIsInvalid: str
        StreetAddressIsRequired: str
        StreetAddressIsTooLong: str
        TooManyAddressesPerCustomer: str

    class ApplePay:
        ApplePayCardsAreNotAccepted: str
        CustomerIdIsRequiredForVaulting: str
        TokenIsInUse: str
        PaymentMethodNonceConsumed: str
        PaymentMethodNonceUnknown: str
        PaymentMethodNonceLocked: str
        PaymentMethodNonceCardTypeIsNotAccepted: str
        CannotUpdateApplePayCardUsingPaymentMethodNonce: str
        NumberIsRequired: str
        ExpirationMonthIsRequired: str
        ExpirationYearIsRequired: str
        CryptogramIsRequired: str
        DecryptionFailed: str
        Disabled: str
        MerchantNotConfigured: str
        MerchantKeysAlreadyConfigured: str
        MerchantKeysNotConfigured: str
        CertificateInvalid: str
        CertificateMismatch: str
        InvalidToken: str
        PrivateKeyMismatch: str
        KeyMismatchStoringCertificate: str

    class AuthorizationFingerprint:
        MissingFingerprint: str
        InvalidFormat: str
        SignatureRevoked: str
        InvalidCreatedAt: str
        InvalidPublicKey: str
        InvalidSignature: str
        OptionsNotAllowedWithoutCustomer: str

    class ClientToken:
        MakeDefaultRequiresCustomerId: str
        VerifyCardRequiresCustomerId: str
        FailOnDuplicatePaymentMethodRequiresCustomerId: str
        CustomerDoesNotExist: str
        ProxyMerchantDoesNotExist: str
        UnsupportedVersion: str
        MerchantAccountDoesNotExist: str

    class CreditCard:
        BillingAddressConflict: str
        BillingAddressFormatIsInvalid: str
        BillingAddressIdIsInvalid: str
        CannotUpdateCardUsingPaymentMethodNonce: str
        CardholderNameIsTooLong: str
        CreditCardTypeIsNotAccepted: str
        CreditCardTypeIsNotAcceptedBySubscriptionMerchantAccount: str
        CustomerIdIsInvalid: str
        CustomerIdIsRequired: str
        CvvIsInvalid: str
        CvvIsRequired: str
        CvvVerificationFailed: str
        DuplicateCardExists: str
        ExpirationDateConflict: str
        ExpirationDateIsInvalid: str
        ExpirationDateIsRequired: str
        ExpirationDateYearIsInvalid: str
        ExpirationMonthIsInvalid: str
        ExpirationYearIsInvalid: str
        InvalidParamsForCreditCardUpdate: str
        InvalidVenmoSDKPaymentMethodCode: str
        NumberHasInvalidLength: str
        NumberLengthIsInvalid: str
        NumberIsInvalid: str
        NumberIsProhibited: str
        NumberIsRequired: str
        NumberMustBeTestNumber: str
        PaymentMethodConflict: str
        PaymentMethodIsNotACreditCard: str
        PaymentMethodNonceCardTypeIsNotAccepted: str
        PaymentMethodNonceConsumed: str
        PaymentMethodNonceLocked: str
        PaymentMethodNonceUnknown: str
        PostalCodeVerificationFailed: str
        TokenInvalid: str
        TokenFormatIsInvalid: str
        TokenIsInUse: str
        TokenIsNotAllowed: str
        TokenIsRequired: str
        TokenIsTooLong: str
        VenmoSDKPaymentMethodCodeCardTypeIsNotAccepted: str
        VerificationNotSupportedOnThisMerchantAccount: str
        VerificationAccountTypeIsInvald: str
        VerificationAccountTypeNotSupported: str

        class Options:
            UpdateExistingTokenIsInvalid: str
            UpdateExistingTokenNotAllowed: str
            VerificationAmountCannotBeNegative: str
            VerificationAmountFormatIsInvalid: str
            VerificationAmountIsTooLarge: str
            VerificationAmountNotSupportedByProcessor: str
            VerificationMerchantAccountIdIsInvalid: str
            VerificationMerchantAccountIsForbidden: str
            VerificationMerchantAccountIsSuspended: str
            VerificationMerchantAccountCannotBeSubMerchantAccount: str

    class Customer:
        CompanyIsTooLong: str
        CustomFieldIsInvalid: str
        CustomFieldIsTooLong: str
        EmailIsInvalid: str
        EmailFormatIsInvalid: str
        EmailIsRequired: str
        EmailIsTooLong: str
        FaxIsTooLong: str
        FirstNameIsTooLong: str
        IdIsInUse: str
        IdIsInvalid: str
        IdIsNotAllowed: str
        IdIsRequired: str
        IdIsTooLong: str
        LastNameIsTooLong: str
        PhoneIsTooLong: str
        VaultedPaymentInstrumentNonceBelongsToDifferentCustomer: str
        WebsiteIsInvalid: str
        WebsiteFormatIsInvalid: str
        WebsiteIsTooLong: str

    class Descriptor:
        DynamicDescriptorsDisabled: str
        InternationalNameFormatIsInvalid: str
        InternationalPhoneFormatIsInvalid: str
        NameFormatIsInvalid: str
        PhoneFormatIsInvalid: str
        UrlFormatIsInvalid: str

    class Dispute:
        CanOnlyAddEvidenceToOpenDispute: str
        CanOnlyRemoveEvidenceFromOpenDispute: str
        CanOnlyAddEvidenceDocumentToDispute: str
        CanOnlyAcceptOpenDispute: str
        CanOnlyFinalizeOpenDispute: str
        CanOnlyCreateEvidenceWithValidCategory: str
        EvidenceContentDateInvalid: str
        EvidenceContentTooLong: str
        EvidenceContentARNTooLong: str
        EvidenceContentPhoneTooLong: str
        EvidenceCategoryTextOnly: str
        EvidenceCategoryDocumentOnly: str
        EvidenceCategoryNotForReasonCode: str
        EvidenceCategoryDuplicate: str
        EvidenceContentEmailInvalid: str
        DigitalGoodsMissingEvidence: str
        DigitalGoodsMissingDownloadDate: str
        NonDisputedPriorTransactionEvidenceMissingARN: str
        NonDisputedPriorTransactionEvidenceMissingDate: str
        RecurringTransactionEvidenceMissingDate: str
        RecurringTransactionEvidenceMissingARN: str
        ValidEvidenceRequiredToFinalize: str

    class DocumentUpload:
        KindIsInvalid: str
        FileIsTooLarge: str
        FileTypeIsInvalid: str
        FileIsMalformedOrEncrypted: str
        FileIsTooLong: str
        FileIsEmpty: str

    class Merchant:
        CountryCannotBeBlank: str
        CountryCodeAlpha2IsInvalid: str
        CountryCodeAlpha2IsNotAccepted: str
        CountryCodeAlpha3IsInvalid: str
        CountryCodeAlpha3IsNotAccepted: str
        CountryCodeNumericIsInvalid: str
        CountryCodeNumericIsNotAccepted: str
        CountryNameIsInvalid: str
        CountryNameIsNotAccepted: str
        CurrenciesAreInvalid: str
        EmailFormatIsInvalid: str
        EmailIsRequired: str
        InconsistentCountry: str
        PaymentMethodsAreInvalid: str
        PaymentMethodsAreNotAllowed: str
        MerchantAccountExistsForCurrency: str
        CurrencyIsRequired: str
        CurrencyIsInvalid: str
        NoMerchantAccounts: str
        MerchantAccountExistsForId: str

    class MerchantAccount:
        IdFormatIsInvalid: str
        IdIsInUse: str
        IdIsNotAllowed: str
        IdIsTooLong: str
        MasterMerchantAccountIdIsInvalid: str
        MasterMerchantAccountIdIsRequired: str
        MasterMerchantAccountMustBeActive: str
        TosAcceptedIsRequired: str
        CannotBeUpdated: str
        IdCannotBeUpdated: str
        MasterMerchantAccountIdCannotBeUpdated: str
        Declined: str
        DeclinedMasterCardMatch: str
        DeclinedOFAC: str
        DeclinedFailedKYC: str
        DeclinedSsnInvalid: str
        DeclinedSsnMatchesDeceased: str

        class ApplicantDetails:
            AccountNumberIsRequired: str
            CompanyNameIsInvalid: str
            CompanyNameIsRequiredWithTaxId: str
            DateOfBirthIsRequired: str
            Declined: str
            DeclinedMasterCardMatch: str
            DeclinedOFAC: str
            DeclinedFailedKYC: str
            DeclinedSsnInvalid: str
            DeclinedSsnMatchesDeceased: str
            EmailAddressIsInvalid: str
            FirstNameIsInvalid: str
            FirstNameIsRequired: str
            LastNameIsInvalid: str
            LastNameIsRequired: str
            PhoneIsInvalid: str
            RoutingNumberIsInvalid: str
            RoutingNumberIsRequired: str
            SsnIsInvalid: str
            TaxIdIsInvalid: str
            TaxIdIsRequiredWithCompanyName: str
            DateOfBirthIsInvalid: str
            EmailAddressIsRequired: str
            AccountNumberIsInvalid: str
            TaxIdMustBeBlank: str

            class Address:
                LocalityIsRequired: str
                PostalCodeIsInvalid: str
                PostalCodeIsRequired: str
                RegionIsRequired: str
                StreetAddressIsInvalid: str
                StreetAddressIsRequired: str
                RegionIsInvalid: str

        class Individual:
            FirstNameIsRequired: str
            LastNameIsRequired: str
            DateOfBirthIsRequired: str
            SsnIsInvalid: str
            EmailAddressIsInvalid: str
            FirstNameIsInvalid: str
            LastNameIsInvalid: str
            PhoneIsInvalid: str
            DateOfBirthIsInvalid: str
            EmailAddressIsRequired: str

            class Address:
                StreetAddressIsRequired: str
                LocalityIsRequired: str
                PostalCodeIsRequired: str
                RegionIsRequired: str
                StreetAddressIsInvalid: str
                PostalCodeIsInvalid: str
                RegionIsInvalid: str

        class Business:
            DbaNameIsInvalid: str
            LegalNameIsInvalid: str
            LegalNameIsRequiredWithTaxId: str
            TaxIdIsInvalid: str
            TaxIdIsRequiredWithLegalName: str
            TaxIdMustBeBlank: str

            class Address:
                StreetAddressIsInvalid: str
                PostalCodeIsInvalid: str
                RegionIsInvalid: str

        class Funding:
            RoutingNumberIsRequired: str
            AccountNumberIsRequired: str
            RoutingNumberIsInvalid: str
            AccountNumberIsInvalid: str
            DestinationIsInvalid: str
            DestinationIsRequired: str
            EmailAddressIsInvalid: str
            EmailAddressIsRequired: str
            MobilePhoneIsInvalid: str
            MobilePhoneIsRequired: str

    class OAuth:
        InvalidGrant: str
        InvalidCredentials: str
        InvalidScope: str
        InvalidRequest: str
        UnsupportedGrantType: str

    class Verification:
        ThreeDSecureAuthenticationIdIsInvalid: str
        ThreeDSecureAuthenticationIdDoesntMatchNonceThreeDSecureAuthentication: str
        ThreeDSecureTransactionPaymentMethodDoesntMatchThreeDSecureAuthenticationPaymentMethod: str
        ThreeDSecureAuthenticationIdWithThreeDSecurePassThruIsInvalid: str
        ThreeDSecureAuthenticationFailed: str
        ThreeDSecureTokenIsInvalid: str
        ThreeDSecureVerificationDataDoesntMatchVerify: str
        MerchantAccountDoesNotSupport3DSecure: str
        MerchantAcountDoesNotMatch3DSecureMerchantAccount: str
        AmountDoesNotMatch3DSecureAmount: str

        class ThreeDSecurePassThru:
            EciFlagIsRequired: str
            EciFlagIsInvalid: str
            CavvIsRequired: str
            ThreeDSecureVersionIsRequired: str
            ThreeDSecureVersionIsInvalid: str
            AuthenticationResponseIsInvalid: str
            DirectoryResponseIsInvalid: str
            CavvAlgorithmIsInvalid: str

        class Options:
            AmountCannotBeNegative: str
            AmountFormatIsInvalid: str
            AmountIsTooLarge: str
            AmountNotSupportedByProcessor: str
            MerchantAccountIdIsInvalid: str
            MerchantAccountIsSuspended: str
            MerchantAccountIsForbidden: str
            MerchantAccountCannotBeSubMerchantAccount: str
            AccountTypeIsInvalid: str
            AccountTypeNotSupported: str

    class PaymentMethod:
        CannotForwardPaymentMethodType: str
        PaymentMethodParamsAreRequired: str
        NonceIsInvalid: str
        NonceIsRequired: str
        CustomerIdIsRequired: str
        CustomerIdIsInvalid: str
        PaymentMethodNonceConsumed: str
        PaymentMethodNonceUnknown: str
        PaymentMethodNonceLocked: str
        PaymentMethodNoLongerSupported: str
        AuthExpired: str
        CannotHaveFundingSourceWithoutAccessToken: str
        InvalidFundingSourceSelection: str
        CannotUpdatePayPalAccountUsingPaymentMethodNonce: str

        class Options:
            UsBankAccountVerificationMethodIsInvalid: str

    class PayPalAccount:
        CannotHaveBothAccessTokenAndConsentCode: str
        CannotVaultOneTimeUsePayPalAccount: str
        ConsentCodeOrAccessTokenIsRequired: str
        CustomerIdIsRequiredForVaulting: str
        InvalidParamsForPayPalAccountUpdate: str
        PayPalAccountsAreNotAccepted: str
        PayPalCommunicationError: str
        PaymentMethodNonceConsumed: str
        PaymentMethodNonceLocked: str
        PaymentMethodNonceUnknown: str
        TokenIsInUse: str

    class SettlementBatchSummary:
        CustomFieldIsInvalid: str
        SettlementDateIsInvalid: str
        SettlementDateIsRequired: str

    class SEPAMandate:
        TypeIsRequired: str
        IBANInvalidCharacter: str
        BICInvalidCharacter: str
        BICLengthIsInvalid: str
        BICUnsupportedCountry: str
        IBANUnsupportedCountry: str
        IBANInvalidFormat: str
        BillingAddressConflict: str
        BillingAddressIdIsInvalid: str
        TypeIsInvalid: str

    class EuropeBankAccount:
        BICIsRequired: str
        IBANIsRequired: str
        AccountHolderNameIsRequired: str

    class Subscription:
        BillingDayOfMonthCannotBeUpdated: str
        BillingDayOfMonthIsInvalid: str
        BillingDayOfMonthMustBeNumeric: str
        CannotAddDuplicateAddonOrDiscount: str
        CannotEditCanceledSubscription: str
        CannotEditExpiredSubscription: str
        CannotEditPriceChangingFieldsOnPastDueSubscription: str
        FirstBillingDateCannotBeInThePast: str
        FirstBillingDateCannotBeUpdated: str
        FirstBillingDateIsInvalid: str
        IdIsInUse: str
        InconsistentNumberOfBillingCycles: str
        InconsistentStartDate: str
        InvalidRequestFormat: str
        MerchantAccountDoesNotSupportInstrumentType: str
        MerchantAccountIdIsInvalid: str
        MismatchCurrencyISOCode: str
        NumberOfBillingCyclesCannotBeBlank: str
        NumberOfBillingCyclesIsTooSmall: str
        NumberOfBillingCyclesMustBeGreaterThanZero: str
        NumberOfBillingCyclesMustBeNumeric: str
        PaymentMethodNonceCardTypeIsNotAccepted: str
        PaymentMethodNonceInstrumentTypeDoesNotSupportSubscriptions: str
        PaymentMethodNonceIsInvalid: str
        PaymentMethodNonceNotAssociatedWithCustomer: str
        PaymentMethodNonceUnvaultedCardIsNotAccepted: str
        PaymentMethodTokenCardTypeIsNotAccepted: str
        PaymentMethodTokenInstrumentTypeDoesNotSupportSubscriptions: str
        PaymentMethodTokenIsInvalid: str
        PaymentMethodTokenNotAssociatedWithCustomer: str
        PlanBillingFrequencyCannotBeUpdated: str
        PlanIdIsInvalid: str
        PriceCannotBeBlank: str
        PriceFormatIsInvalid: str
        PriceIsTooLarge: str
        StatusIsCanceled: str
        TokenFormatIsInvalid: str
        TrialDurationFormatIsInvalid: str
        TrialDurationIsRequired: str
        TrialDurationUnitIsInvalid: str

        class Modification:
            AmountCannotBeBlank: str
            AmountIsInvalid: str
            AmountIsTooLarge: str
            CannotEditModificationsOnPastDueSubscription: str
            CannotUpdateAndRemove: str
            ExistingIdIsIncorrectKind: str
            ExistingIdIsInvalid: str
            ExistingIdIsRequired: str
            IdToRemoveIsIncorrectKind: str
            IdToRemoveIsNotPresent: str
            InconsistentNumberOfBillingCycles: str
            InheritedFromIdIsInvalid: str
            InheritedFromIdIsRequired: str
            Missing: str
            NumberOfBillingCyclesCannotBeBlank: str
            NumberOfBillingCyclesIsInvalid: str
            NumberOfBillingCyclesMustBeGreaterThanZero: str
            QuantityCannotBeBlank: str
            QuantityIsInvalid: str
            QuantityMustBeGreaterThanZero: str
            IdToRemoveIsInvalid: str

    class Transaction:
        AdjustmentAmountMustBeGreaterThanZero: str
        AmountCannotBeNegative: str
        AmountDoesNotMatch3DSecureAmount: str
        AmountIsInvalid: str
        AmountFormatIsInvalid: str
        AmountIsRequired: str
        AmountIsTooLarge: str
        AmountMustBeGreaterThanZero: str
        AmountNotSupportedByProcessor: str
        BillingAddressConflict: str
        BillingPhoneNumberIsInvalid: str
        CannotBeVoided: str
        CannotCancelRelease: str
        CannotCloneCredit: str
        CannotCloneMarketplaceTransaction: str
        CannotCloneTransactionWithPayPalAccount: str
        CannotCloneTransactionWithVaultCreditCard: str
        CannotCloneUnsuccessfulTransaction: str
        CannotCloneVoiceAuthorizations: str
        CannotHoldInEscrow: str
        CannotPartiallyRefundEscrowedTransaction: str
        CannotRefundCredit: str
        CannotRefundSettlingTransaction: str
        CannotRefundUnlessSettled: str
        CannotRefundWithPendingMerchantAccount: str
        CannotRefundWithSuspendedMerchantAccount: str
        CannotReleaseFromEscrow: str
        CannotSimulateTransactionSettlement: str
        CannotSubmitForPartialSettlement: str
        CannotSubmitForSettlement: str
        CannotUpdateTransactionDetailsNotSubmittedForSettlement: str
        ChannelIsTooLong: str
        CreditCardIsRequired: str
        CustomFieldIsInvalid: str
        CustomFieldIsTooLong: str
        CustomerDefaultPaymentMethodCardTypeIsNotAccepted: str
        CustomerDoesNotHaveCreditCard: str
        CustomerIdIsInvalid: str
        DiscountAmountCannotBeNegative: str
        DiscountAmountFormatIsInvalid: str
        DiscountAmountIsTooLarge: str
        ExchangeRateQuoteIdIsTooLong: str
        FailedAuthAdjustmentAllowRetry: str
        FailedAuthAdjustmentHardDecline: str
        FinalAuthSubmitForSettlementForDifferentAmount: str
        HasAlreadyBeenRefunded: str
        LineItemsExpected: str
        MerchantAccountDoesNotMatch3DSecureMerchantAccount: str
        MerchantAccountDoesNotSupportMOTO: str
        MerchantAccountDoesNotSupportRefunds: str
        MerchantAccountIdDoesNotMatchSubscription: str
        MerchantAccountIdIsInvalid: str
        MerchantAccountIsSuspended: str
        NoNetAmountToPerformAuthAdjustment: str
        OrderIdIsTooLong: str
        PayPalAuthExpired: str
        PayPalNotEnabled: str
        PayPalVaultRecordMissingData: str
        PaymentInstrumentNotSupportedByMerchantAccount: str
        PaymentInstrumentTypeIsNotAccepted: str
        PaymentInstrumentWithExternalVaultIsInvalid: str
        PaymentMethodConflict: str
        PaymentMethodConflictWithVenmoSDK: str
        PaymentMethodDoesNotBelongToCustomer: str
        PaymentMethodDoesNotBelongToSubscription: str
        PaymentMethodNonceCardTypeIsNotAccepted: str
        PaymentMethodNonceConsumed: str
        PaymentMethodNonceHasNoValidPaymentInstrumentType: str
        PaymentMethodNonceLocked: str
        PaymentMethodNonceUnknown: str
        PaymentMethodTokenCardTypeIsNotAccepted: str
        PaymentMethodTokenIsInvalid: str
        ProcessorAuthorizationCodeCannotBeSet: str
        ProcessorAuthorizationCodeIsInvalid: str
        ProcessorDoesNotSupportAuths: str
        ProcessorDoesNotSupportAuthAdjustment: str
        ProcessorDoesNotSupportCredits: str
        ProcessorDoesNotSupportIncrementalAuth: str
        ProcessorDoesNotSupportMotoForCardType: str
        ProcessorDoesNotSupportPartialAuthReversal: str
        ProcessorDoesNotSupportPartialSettlement: str
        ProcessorDoesNotSupportUpdatingDescriptor: str
        ProcessorDoesNotSupportUpdatingOrderId: str
        ProcessorDoesNotSupportUpdatingTransactionDetails: str
        ProcessorDoesNotSupportVoiceAuthorizations: str
        ProductSkuIsInvalid: str
        PurchaseOrderNumberIsInvalid: str
        PurchaseOrderNumberIsTooLong: str
        RefundAmountIsTooLarge: str
        RefundAuthHardDeclined: str
        RefundAuthSoftDeclined: str
        ScaExemptionInvalid: str
        ServiceFeeAmountCannotBeNegative: str
        ServiceFeeAmountFormatIsInvalid: str
        ServiceFeeAmountIsTooLarge: str
        ServiceFeeAmountNotAllowedOnMasterMerchantAccount: str
        ServiceFeeIsNotAllowedOnCredits: str
        ServiceFeeNotAcceptedForPayPal: str
        SettlementAmountIsLessThanServiceFeeAmount: str
        SettlementAmountIsTooLarge: str
        ShippingAddressDoesntMatchCustomer: str
        ShippingAmountCannotBeNegative: str
        ShippingAmountFormatIsInvalid: str
        ShippingAmountIsTooLarge: str
        ShippingMethodIsInvalid: str
        ShippingPhoneNumberIsInvalid: str
        ShipsFromPostalCodeInvalidCharacters: str
        ShipsFromPostalCodeIsInvalid: str
        ShipsFromPostalCodeIsTooLong: str
        SubMerchantAccountRequiresServiceFeeAmount: str
        SubscriptionDoesNotBelongToCustomer: str
        SubscriptionIdIsInvalid: str
        SubscriptionStatusMustBePastDue: str
        TaxAmountCannotBeNegative: str
        TaxAmountFormatIsInvalid: str
        TaxAmountIsRequiredForAibSwedish: str
        TaxAmountIsTooLarge: str
        ThreeDSecureAuthenticationFailed: str
        ThreeDSecureAuthenticationIdDoesntMatchNonceThreeDSecureAuthentication: str
        ThreeDSecureAuthenticationIdIsInvalid: str
        ThreeDSecureAuthenticationIdWithThreeDSecurePassThruIsInvalid: str
        ThreeDSecureAuthenticationResponseIsInvalid: str
        ThreeDSecureCavvAlgorithmIsInvalid: str
        ThreeDSecureCavvIsRequired: str
        ThreeDSecureDirectoryResponseIsInvalid: str
        ThreeDSecureEciFlagIsInvalid: str
        ThreeDSecureEciFlagIsRequired: str
        ThreeDSecureMerchantAccountDoesNotSupportCardType: str
        ThreeDSecureTokenIsInvalid: str
        ThreeDSecureTransactionDataDoesntMatchVerify: str
        ThreeDSecureTransactionPaymentMethodDoesntMatchThreeDSecureAuthenticationPaymentMethod: str
        ThreeDSecureXidIsRequired: str
        TooManyLineItems: str
        TransactionIsNotEligibleForAdjustment: str
        TransactionMustBeInStateAuthorized: str
        TransactionSourceIsInvalid: str
        TypeIsInvalid: str
        TypeIsRequired: str
        UnsupportedVoiceAuthorization: str
        UsBankAccountNonceMustBePlaidVerified: str
        UsBankAccountNotVerified: str

        class ExternalVault:
            StatusIsInvalid: str
            StatusWithPreviousNetworkTransactionIdIsInvalid: str
            CardTypeIsInvalid: str
            PreviousNetworkTransactionIdIsInvalid: str

        class Options:
            SubmitForSettlementIsRequiredForCloning: str
            SubmitForSettlementIsRequiredForPayPalUnilateral: str
            UseBillingForShippingDisabled: str
            VaultIsDisabled: str

            class PayPal:
                CustomFieldTooLong: str

            class CreditCard:
                AccountTypeIsInvalid: str
                AccountTypeNotSupported: str
                AccountTypeDebitDoesNotSupportAuths: str

        class Industry:
            IndustryTypeIsInvalid: str

            class Lodging:
                EmptyData: str
                FolioNumberIsInvalid: str
                CheckInDateIsInvalid: str
                CheckOutDateIsInvalid: str
                CheckOutDateMustFollowCheckInDate: str
                UnknownDataField: str
                RoomRateMustBeGreaterThanZero: str
                RoomRateFormatIsInvalid: str
                RoomRateIsTooLarge: str
                RoomTaxMustBeGreaterThanZero: str
                RoomTaxFormatIsInvalid: str
                RoomTaxIsTooLarge: str
                NoShowIndicatorIsInvalid: str
                AdvancedDepositIndicatorIsInvalid: str
                FireSafetyIndicatorIsInvalid: str
                PropertyPhoneIsInvalid: str

            class TravelCruise:
                EmptyData: str
                UnknownDataField: str
                TravelPackageIsInvalid: str
                DepartureDateIsInvalid: str
                LodgingCheckInDateIsInvalid: str
                LodgingCheckOutDateIsInvalid: str

            class TravelFlight:
                EmptyData: str
                UnknownDataField: str
                CustomerCodeIsTooLong: str
                FareAmountCannotBeNegative: str
                FareAmountFormatIsInvalid: str
                FareAmountIsTooLarge: str
                FeeAmountCannotBeNegative: str
                FeeAmountFormatIsInvalid: str
                FeeAmountIsTooLarge: str
                IssuedDateFormatIsInvalid: str
                IssuingCarrierCodeIsTooLong: str
                PassengerMiddleInitialIsTooLong: str
                RestrictedTicketIsRequired: str
                TaxAmountCannotBeNegative: str
                TaxAmountFormatIsInvalid: str
                TaxAmountIsTooLarge: str
                TicketNumberIsTooLong: str
                LegsExpected: str
                TooManyLegs: str

            class Leg:
                class TravelFlight:
                    ArrivalAirportCodeIsTooLong: str
                    ArrivalTimeFormatIsInvalid: str
                    CarrierCodeIsTooLong: str
                    ConjunctionTicketIsTooLong: str
                    CouponNumberIsTooLong: str
                    DepartureAirportCodeIsTooLong: str
                    DepartureTimeFormatIsInvalid: str
                    ExchangeTicketIsTooLong: str
                    FareAmountCannotBeNegative: str
                    FareAmountFormatIsInvalid: str
                    FareAmountIsTooLarge: str
                    FareBasisCodeIsTooLong: str
                    FeeAmountCannotBeNegative: str
                    FeeAmountFormatIsInvalid: str
                    FeeAmountIsTooLarge: str
                    ServiceClassIsTooLong: str
                    TaxAmountCannotBeNegative: str
                    TaxAmountFormatIsInvalid: str
                    TaxAmountIsTooLarge: str
                    TicketNumberIsTooLong: str

            class AdditionalCharge:
                KindIsInvalid: str
                KindMustBeUnique: str
                AmountMustBeGreaterThanZero: str
                AmountFormatIsInvalid: str
                AmountIsTooLarge: str
                AmountIsRequired: str

        class LineItem:
            CommodityCodeIsTooLong: str
            DescriptionIsTooLong: str
            DiscountAmountFormatIsInvalid: str
            DiscountAmountIsTooLarge: str
            DiscountAmountCannotBeNegative: str
            KindIsInvalid: str
            KindIsRequired: str
            NameIsRequired: str
            NameIsTooLong: str
            ProductCodeIsTooLong: str
            QuantityFormatIsInvalid: str
            QuantityIsRequired: str
            QuantityIsTooLarge: str
            TotalAmountFormatIsInvalid: str
            TotalAmountIsRequired: str
            TotalAmountIsTooLarge: str
            TotalAmountMustBeGreaterThanZero: str
            UnitAmountFormatIsInvalid: str
            UnitAmountIsRequired: str
            UnitAmountIsTooLarge: str
            UnitAmountMustBeGreaterThanZero: str
            UnitOfMeasureIsTooLarge: str
            UnitTaxAmountFormatIsInvalid: str
            UnitTaxAmountIsTooLarge: str
            UnitTaxAmountCannotBeNegative: str
            TaxAmountFormatIsInvalid: str
            TaxAmountIsTooLarge: str
            TaxAmountCannotBeNegative: str

    class UsBankAccountVerification:
        NotConfirmable: str
        MustBeMicroTransfersVerification: str
        AmountsDoNotMatch: str
        TooManyConfirmationAttempts: str
        UnableToConfirmDepositAmounts: str
        InvalidDepositAmounts: str

    class RiskData:
        CustomerBrowserIsTooLong: str
        CustomerDeviceIdIsTooLong: str
        CustomerLocationZipInvalidCharacters: str
        CustomerLocationZipIsInvalid: str
        CustomerLocationZipIsTooLong: str
        CustomerTenureIsTooLong: str
