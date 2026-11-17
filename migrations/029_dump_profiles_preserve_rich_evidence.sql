-- Migration 029: dump the live "never shrink" trigger on public.profiles.
--
-- NO BEHAVIOUR CHANGE. This file records, word for word, what is live on the
-- shared Supabase database (project linkedin-enricher) on 2026-11-17, so the
-- next migration (030, the profile replace function) has a reviewed base to
-- change. Until now this trigger existed only in the database, not in git.
--
-- Source of truth:
--   select pg_get_functiondef('sourcing_core.profiles_preserve_rich_evidence'::regproc);
--   select pg_get_triggerdef(oid) from pg_trigger
--    where tgrelid = 'public.profiles'::regclass
--      and tgname = 'trg_profiles_preserve_rich_evidence';
--
-- Applying this file is a no-op: CREATE OR REPLACE with the identical body
-- keeps the function's owner and grants, and the trigger is dropped and
-- re-created with the same name, timing, event and function (BEFORE triggers
-- fire in name order, so the firing order is unchanged too). Both run inside
-- one transaction when applied as a migration.
--
-- What the function does (for readers):
--   (a) raw_data: an update that makes the JSON text shorter, or null, is
--       reverted to the old value.
--   (b) skills / all_titles / all_employers / all_schools: an update that
--       makes the array shorter is reverted to the old value.
--   (c) name, current_title, current_company, location, skills_blob,
--       titles_blob, schema_version, github_url, normalized_title,
--       seniority_level: a blank or null new value keeps the old value.

-- BEGIN LIVE FUNCTION DEFINITION
CREATE OR REPLACE FUNCTION sourcing_core.profiles_preserve_rich_evidence()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO ''
AS $function$
begin
  if old.raw_data is not null and (
    new.raw_data is null
    or octet_length(new.raw_data::text) < octet_length(old.raw_data::text)
  ) then
    new.raw_data := old.raw_data;
  end if;

  if coalesce(cardinality(new.skills), 0) < coalesce(cardinality(old.skills), 0) then
    new.skills := old.skills;
  end if;
  if coalesce(cardinality(new.all_titles), 0) < coalesce(cardinality(old.all_titles), 0) then
    new.all_titles := old.all_titles;
  end if;
  if coalesce(cardinality(new.all_employers), 0) < coalesce(cardinality(old.all_employers), 0) then
    new.all_employers := old.all_employers;
  end if;
  if coalesce(cardinality(new.all_schools), 0) < coalesce(cardinality(old.all_schools), 0) then
    new.all_schools := old.all_schools;
  end if;

  new.name := coalesce(nullif(btrim(new.name), ''), old.name);
  new.current_title := coalesce(nullif(btrim(new.current_title), ''), old.current_title);
  new.current_company := coalesce(nullif(btrim(new.current_company), ''), old.current_company);
  new.location := coalesce(nullif(btrim(new.location), ''), old.location);
  new.skills_blob := coalesce(nullif(btrim(new.skills_blob), ''), old.skills_blob);
  new.titles_blob := coalesce(nullif(btrim(new.titles_blob), ''), old.titles_blob);
  new.schema_version := coalesce(nullif(btrim(new.schema_version), ''), old.schema_version);
  new.github_url := coalesce(nullif(btrim(new.github_url), ''), old.github_url);
  new.normalized_title := coalesce(nullif(btrim(new.normalized_title), ''), old.normalized_title);
  new.seniority_level := coalesce(nullif(btrim(new.seniority_level), ''), old.seniority_level);

  return new;
end;
$function$
;
-- END LIVE FUNCTION DEFINITION

-- Live trigger definition (pg_get_triggerdef), re-created unchanged:
DROP TRIGGER IF EXISTS trg_profiles_preserve_rich_evidence ON public.profiles;
CREATE TRIGGER trg_profiles_preserve_rich_evidence BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE FUNCTION sourcing_core.profiles_preserve_rich_evidence();
