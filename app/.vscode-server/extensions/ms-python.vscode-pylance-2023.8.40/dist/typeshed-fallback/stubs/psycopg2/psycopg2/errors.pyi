from psycopg2._psycopg import Error as Error, Warning as Warning

class DatabaseError(Error): ...
class InterfaceError(Error): ...
class DataError(DatabaseError): ...
class DiagnosticsException(DatabaseError): ...
class IntegrityError(DatabaseError): ...
class InternalError(DatabaseError): ...
class InvalidGrantOperation(DatabaseError): ...
class InvalidGrantor(DatabaseError): ...
class InvalidLocatorSpecification(DatabaseError): ...
class InvalidRoleSpecification(DatabaseError): ...
class InvalidTransactionInitiation(DatabaseError): ...
class LocatorException(DatabaseError): ...
class NoAdditionalDynamicResultSetsReturned(DatabaseError): ...
class NoData(DatabaseError): ...
class NotSupportedError(DatabaseError): ...
class OperationalError(DatabaseError): ...
class ProgrammingError(DatabaseError): ...
class SnapshotTooOld(DatabaseError): ...
class SqlStatementNotYetComplete(DatabaseError): ...
class StackedDiagnosticsAccessedWithoutActiveHandler(DatabaseError): ...
class TriggeredActionException(DatabaseError): ...
class ActiveSqlTransaction(InternalError): ...
class AdminShutdown(OperationalError): ...
class AmbiguousAlias(ProgrammingError): ...
class AmbiguousColumn(ProgrammingError): ...
class AmbiguousFunction(ProgrammingError): ...
class AmbiguousParameter(ProgrammingError): ...
class ArraySubscriptError(DataError): ...
class AssertFailure(InternalError): ...
class BadCopyFileFormat(DataError): ...
class BranchTransactionAlreadyActive(InternalError): ...
class CannotCoerce(ProgrammingError): ...
class CannotConnectNow(OperationalError): ...
class CantChangeRuntimeParam(OperationalError): ...
class CardinalityViolation(ProgrammingError): ...
class CaseNotFound(ProgrammingError): ...
class CharacterNotInRepertoire(DataError): ...
class CheckViolation(IntegrityError): ...
class CollationMismatch(ProgrammingError): ...
class ConfigFileError(InternalError): ...
class ConfigurationLimitExceeded(OperationalError): ...
class ConnectionDoesNotExist(OperationalError): ...
class ConnectionException(OperationalError): ...
class ConnectionFailure(OperationalError): ...
class ContainingSqlNotPermitted(InternalError): ...
class CrashShutdown(OperationalError): ...
class DataCorrupted(InternalError): ...
class DataException(DataError): ...
class DatabaseDropped(OperationalError): ...
class DatatypeMismatch(ProgrammingError): ...
class DatetimeFieldOverflow(DataError): ...
class DependentObjectsStillExist(InternalError): ...
class DependentPrivilegeDescriptorsStillExist(InternalError): ...
class DiskFull(OperationalError): ...
class DivisionByZero(DataError): ...
class DuplicateAlias(ProgrammingError): ...
class DuplicateColumn(ProgrammingError): ...
class DuplicateCursor(ProgrammingError): ...
class DuplicateDatabase(ProgrammingError): ...
class DuplicateFile(OperationalError): ...
class DuplicateFunction(ProgrammingError): ...
class DuplicateJsonObjectKeyValue(DataError): ...
class DuplicateObject(ProgrammingError): ...
class DuplicatePreparedStatement(ProgrammingError): ...
class DuplicateSchema(ProgrammingError): ...
class DuplicateTable(ProgrammingError): ...
class ErrorInAssignment(DataError): ...
class EscapeCharacterConflict(DataError): ...
class EventTriggerProtocolViolated(InternalError): ...
class ExclusionViolation(IntegrityError): ...
class ExternalRoutineException(InternalError): ...
class ExternalRoutineInvocationException(InternalError): ...
class FdwColumnNameNotFound(OperationalError): ...
class FdwDynamicParameterValueNeeded(OperationalError): ...
class FdwError(OperationalError): ...
class FdwFunctionSequenceError(OperationalError): ...
class FdwInconsistentDescriptorInformation(OperationalError): ...
class FdwInvalidAttributeValue(OperationalError): ...
class FdwInvalidColumnName(OperationalError): ...
class FdwInvalidColumnNumber(OperationalError): ...
class FdwInvalidDataType(OperationalError): ...
class FdwInvalidDataTypeDescriptors(OperationalError): ...
class FdwInvalidDescriptorFieldIdentifier(OperationalError): ...
class FdwInvalidHandle(OperationalError): ...
class FdwInvalidOptionIndex(OperationalError): ...
class FdwInvalidOptionName(OperationalError): ...
class FdwInvalidStringFormat(OperationalError): ...
class FdwInvalidStringLengthOrBufferLength(OperationalError): ...
class FdwInvalidUseOfNullPointer(OperationalError): ...
class FdwNoSchemas(OperationalError): ...
class FdwOptionNameNotFound(OperationalError): ...
class FdwOutOfMemory(OperationalError): ...
class FdwReplyHandle(OperationalError): ...
class FdwSchemaNotFound(OperationalError): ...
class FdwTableNotFound(OperationalError): ...
class FdwTooManyHandles(OperationalError): ...
class FdwUnableToCreateExecution(OperationalError): ...
class FdwUnableToCreateReply(OperationalError): ...
class FdwUnableToEstablishConnection(OperationalError): ...
class FeatureNotSupported(NotSupportedError): ...
class FloatingPointException(DataError): ...
class ForeignKeyViolation(IntegrityError): ...
class FunctionExecutedNoReturnStatement(InternalError): ...
class GeneratedAlways(ProgrammingError): ...
class GroupingError(ProgrammingError): ...
class HeldCursorRequiresSameIsolationLevel(InternalError): ...
class IdleInTransactionSessionTimeout(InternalError): ...
class InFailedSqlTransaction(InternalError): ...
class InappropriateAccessModeForBranchTransaction(InternalError): ...
class InappropriateIsolationLevelForBranchTransaction(InternalError): ...
class IndeterminateCollation(ProgrammingError): ...
class IndeterminateDatatype(ProgrammingError): ...
class IndexCorrupted(InternalError): ...
class IndicatorOverflow(DataError): ...
class InsufficientPrivilege(ProgrammingError): ...
class InsufficientResources(OperationalError): ...
class IntegrityConstraintViolation(IntegrityError): ...
class InternalError_(InternalError): ...
class IntervalFieldOverflow(DataError): ...
class InvalidArgumentForLogarithm(DataError): ...
class InvalidArgumentForNthValueFunction(DataError): ...
class InvalidArgumentForNtileFunction(DataError): ...
class InvalidArgumentForPowerFunction(DataError): ...
class InvalidArgumentForSqlJsonDatetimeFunction(DataError): ...
class InvalidArgumentForWidthBucketFunction(DataError): ...
class InvalidAuthorizationSpecification(OperationalError): ...
class InvalidBinaryRepresentation(DataError): ...
class InvalidCatalogName(ProgrammingError): ...
class InvalidCharacterValueForCast(DataError): ...
class InvalidColumnDefinition(ProgrammingError): ...
class InvalidColumnReference(ProgrammingError): ...
class InvalidCursorDefinition(ProgrammingError): ...
class InvalidCursorName(OperationalError): ...
class InvalidCursorState(InternalError): ...
class InvalidDatabaseDefinition(ProgrammingError): ...
class InvalidDatetimeFormat(DataError): ...
class InvalidEscapeCharacter(DataError): ...
class InvalidEscapeOctet(DataError): ...
class InvalidEscapeSequence(DataError): ...
class InvalidForeignKey(ProgrammingError): ...
class InvalidFunctionDefinition(ProgrammingError): ...
class InvalidIndicatorParameterValue(DataError): ...
class InvalidJsonText(DataError): ...
class InvalidName(ProgrammingError): ...
class InvalidObjectDefinition(ProgrammingError): ...
class InvalidParameterValue(DataError): ...
class InvalidPassword(OperationalError): ...
class InvalidPrecedingOrFollowingSize(DataError): ...
class InvalidPreparedStatementDefinition(ProgrammingError): ...
class InvalidRecursion(ProgrammingError): ...
class InvalidRegularExpression(DataError): ...
class InvalidRowCountInLimitClause(DataError): ...
class InvalidRowCountInResultOffsetClause(DataError): ...
class InvalidSavepointSpecification(InternalError): ...
class InvalidSchemaDefinition(ProgrammingError): ...
class InvalidSchemaName(ProgrammingError): ...
class InvalidSqlJsonSubscript(DataError): ...
class InvalidSqlStatementName(OperationalError): ...
class InvalidSqlstateReturned(InternalError): ...
class InvalidTableDefinition(ProgrammingError): ...
class InvalidTablesampleArgument(DataError): ...
class InvalidTablesampleRepeat(DataError): ...
class InvalidTextRepresentation(DataError): ...
class InvalidTimeZoneDisplacementValue(DataError): ...
class InvalidTransactionState(InternalError): ...
class InvalidTransactionTermination(InternalError): ...
class InvalidUseOfEscapeCharacter(DataError): ...
class InvalidXmlComment(DataError): ...
class InvalidXmlContent(DataError): ...
class InvalidXmlDocument(DataError): ...
class InvalidXmlProcessingInstruction(DataError): ...
class IoError(OperationalError): ...
class LockFileExists(InternalError): ...
class LockNotAvailable(OperationalError): ...
class ModifyingSqlDataNotPermitted(InternalError): ...
class ModifyingSqlDataNotPermittedExt(InternalError): ...
class MoreThanOneSqlJsonItem(DataError): ...
class MostSpecificTypeMismatch(DataError): ...
class NameTooLong(ProgrammingError): ...
class NoActiveSqlTransaction(InternalError): ...
class NoActiveSqlTransactionForBranchTransaction(InternalError): ...
class NoDataFound(InternalError): ...
class NoSqlJsonItem(DataError): ...
class NonNumericSqlJsonItem(DataError): ...
class NonUniqueKeysInAJsonObject(DataError): ...
class NonstandardUseOfEscapeCharacter(DataError): ...
class NotAnXmlDocument(DataError): ...
class NotNullViolation(IntegrityError): ...
class NullValueNoIndicatorParameter(DataError): ...
class NullValueNotAllowed(DataError): ...
class NullValueNotAllowedExt(InternalError): ...
class NumericValueOutOfRange(DataError): ...
class ObjectInUse(OperationalError): ...
class ObjectNotInPrerequisiteState(OperationalError): ...
class OperatorIntervention(OperationalError): ...
class OutOfMemory(OperationalError): ...
class PlpgsqlError(InternalError): ...
class ProgramLimitExceeded(OperationalError): ...
class ProhibitedSqlStatementAttempted(InternalError): ...
class ProhibitedSqlStatementAttemptedExt(InternalError): ...
class ProtocolViolation(OperationalError): ...
class QueryCanceledError(OperationalError): ...
class RaiseException(InternalError): ...
class ReadOnlySqlTransaction(InternalError): ...
class ReadingSqlDataNotPermitted(InternalError): ...
class ReadingSqlDataNotPermittedExt(InternalError): ...
class ReservedName(ProgrammingError): ...
class RestrictViolation(IntegrityError): ...
class SavepointException(InternalError): ...
class SchemaAndDataStatementMixingNotSupported(InternalError): ...
class SequenceGeneratorLimitExceeded(DataError): ...
class SingletonSqlJsonItemRequired(DataError): ...
class SqlJsonArrayNotFound(DataError): ...
class SqlJsonMemberNotFound(DataError): ...
class SqlJsonNumberNotFound(DataError): ...
class SqlJsonObjectNotFound(DataError): ...
class SqlJsonScalarRequired(DataError): ...
class SqlRoutineException(InternalError): ...
class SqlclientUnableToEstablishSqlconnection(OperationalError): ...
class SqlserverRejectedEstablishmentOfSqlconnection(OperationalError): ...
class SrfProtocolViolated(InternalError): ...
class StatementTooComplex(OperationalError): ...
class StringDataLengthMismatch(DataError): ...
class StringDataRightTruncation(DataError): ...
class SubstringError(DataError): ...
class SyntaxError(ProgrammingError): ...
class SyntaxErrorOrAccessRuleViolation(ProgrammingError): ...
class SystemError(OperationalError): ...
class TooManyArguments(OperationalError): ...
class TooManyColumns(OperationalError): ...
class TooManyConnections(OperationalError): ...
class TooManyJsonArrayElements(DataError): ...
class TooManyJsonObjectMembers(DataError): ...
class TooManyRows(InternalError): ...
class TransactionResolutionUnknown(OperationalError): ...
class TransactionRollbackError(OperationalError): ...
class TriggerProtocolViolated(InternalError): ...
class TriggeredDataChangeViolation(OperationalError): ...
class TrimError(DataError): ...
class UndefinedColumn(ProgrammingError): ...
class UndefinedFile(OperationalError): ...
class UndefinedFunction(ProgrammingError): ...
class UndefinedObject(ProgrammingError): ...
class UndefinedParameter(ProgrammingError): ...
class UndefinedTable(ProgrammingError): ...
class UniqueViolation(IntegrityError): ...
class UnsafeNewEnumValueUsage(OperationalError): ...
class UnterminatedCString(DataError): ...
class UntranslatableCharacter(DataError): ...
class WindowingError(ProgrammingError): ...
class WithCheckOptionViolation(ProgrammingError): ...
class WrongObjectType(ProgrammingError): ...
class ZeroLengthCharacterString(DataError): ...
class DeadlockDetected(TransactionRollbackError): ...
class QueryCanceled(QueryCanceledError): ...
class SerializationFailure(TransactionRollbackError): ...
class StatementCompletionUnknown(TransactionRollbackError): ...
class TransactionIntegrityConstraintViolation(TransactionRollbackError): ...
class TransactionRollback(TransactionRollbackError): ...

def lookup(code): ...
