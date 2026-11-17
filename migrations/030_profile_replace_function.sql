-- Migration 030: public.supanova_replace_profile — the ONE writer allowed to
-- shrink a profile, and the matching exception in the never-shrink trigger.
--
-- STATUS: NOT APPLIED. Tested on the live database inside a transaction that
-- was rolled back (evidence in the PR). Apply only after review and
-- Alexey's go-ahead.
--
-- Rollback if ever needed (restores migration 029 exactly):
--   drop function public.supanova_replace_profile(text, timestamptz, jsonb);
--   re-run migrations/029_dump_profiles_preserve_rich_evidence.sql;
--   drop schema profile_replace_private cascade;
--
-- WHY THIS EXISTS
-- Supanova's monthly refresh replaces a job-changer's whole profile with a
-- freshly enriched one. The fresh one is often legitimately smaller (cached
-- enrichment trims data). The trigger sourcing_core.profiles_preserve_rich_evidence
-- (migration 029) silently reverts any update that shrinks raw_data or the
-- skills / all_titles / all_employers / all_schools arrays, so the refresh
-- cannot land. This migration adds one function that may pass those two
-- rules, and keeps every other writer fully protected.
--
-- HOW THE BYPASS WORKS (and why a client cannot forge it)
-- 1. public.supanova_replace_profile (SECURITY DEFINER, owner postgres)
--    makes a random per-call token (gen_random_uuid), stores it in
--    profile_replace_private.replace_tokens together with the current
--    transaction id and the target linkedin_url, sets the transaction-local
--    setting sourcing_core.replace_token to the token, runs ONE update,
--    then deletes the token and clears the setting before it returns.
-- 2. The trigger skips rules (a) and (b) only when that setting matches a
--    token row for the SAME transaction id and the SAME linkedin_url as the
--    row being updated. Rule (c) (never blank name/title/company/...) runs
--    for every writer, this function included.
-- Why it cannot be forged:
--   - Anyone can set the setting, but it only helps if it equals a live
--     token. A token exists only inside the function's own call: it is
--     inserted and deleted in the same transaction and never committed, so
--     no other session can ever see it, and it is never returned.
--   - The token table is in a private schema with no grants to PUBLIC,
--     anon, authenticated or service_role, and has RLS on with no policies.
--     The trigger stays SECURITY INVOKER and only reads that table when the
--     setting is non-empty. So a normal update never touches it, and a
--     non-owner role (service_role over REST, boris_n8n, anon,
--     authenticated) that sets the setting by hand gets "permission denied"
--     and its update fails: it fails closed, it never shrinks.
--   - The owner role postgres can read the table, but finds no matching
--     token, so its plain updates are still reverted. (postgres is a
--     superuser and could drop the trigger anyway; this design does not
--     try to defend against the database owner.)
--   - The token is bound to one transaction id and one linkedin_url, so
--     even code running inside the update could not reuse it for another
--     row or another transaction.
--
-- THE FUNCTION'S CONTRACT
--   public.supanova_replace_profile(p_linkedin_url text,
--                                   p_expected_updated_at timestamptz,
--                                   p_cols jsonb) returns timestamptz
--   - UPDATE only. Never inserts or deletes. Matches linkedin_url exactly.
--   - Optimistic check: updates only if the row's updated_at still equals
--     p_expected_updated_at. 0 rows changed (missing row, or someone else
--     changed it since the caller read it) raises an exception.
--   - p_cols whitelist (any other key raises an exception):
--       raw_data (json object), name, current_title, current_company,
--       location (string or null), current_start_date, enriched_at
--       (timestamp string or null), current_years_at_company (number or
--       null), all_titles, all_employers, all_schools, skills (array of
--       strings).
--     id, linkedin_url, email, email_source, original_url(s), embedding*,
--     search_text and created_at can never be touched.
--   - Keys not present in p_cols keep their current value.
--   - Returns the new updated_at (set by the profiles_updated_at trigger).
--   - EXECUTE: service_role (and the owner) only. Reached over REST as
--     POST /rest/v1/rpc/supanova_replace_profile, same as
--     public.sourcingx_upsert_profiles (migration 025).

-- 1. Private schema and token table ------------------------------------------
create schema if not exists profile_replace_private;
revoke all on schema profile_replace_private from public, anon, authenticated, service_role;

create table if not exists profile_replace_private.replace_tokens (
  token        uuid primary key,
  txid         bigint not null,
  linkedin_url text not null,
  created_at   timestamptz not null default now()
);
alter table profile_replace_private.replace_tokens enable row level security;
revoke all on table profile_replace_private.replace_tokens from public, anon, authenticated, service_role;

-- 2. The never-shrink trigger, with the one exception ------------------------
-- Same as migration 029, except rules (a) and (b) are skipped when the
-- update comes from public.supanova_replace_profile. Rule (c) is unchanged.
create or replace function sourcing_core.profiles_preserve_rich_evidence()
 returns trigger
 language plpgsql
 set search_path to ''
as $function$
declare
  v_token  text;
  v_bypass boolean := false;
begin
  -- Only look at the token table when the setting is non-empty, so ordinary
  -- writers never need access to it.
  v_token := nullif(current_setting('sourcing_core.replace_token', true), '');
  if v_token is not null then
    select exists (
      select 1
        from profile_replace_private.replace_tokens t
       where t.token::text = v_token
         and t.txid = txid_current()
         and t.linkedin_url = old.linkedin_url
    ) into v_bypass;
  end if;

  if not v_bypass then
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
$function$;

-- 3. The replace function ------------------------------------------------------
create or replace function public.supanova_replace_profile(
  p_linkedin_url        text,
  p_expected_updated_at timestamptz,
  p_cols                jsonb
)
returns timestamptz
language plpgsql
security definer
set search_path = ''
as $function$
declare
  c_allowed constant text[] := array[
    'raw_data', 'name', 'current_title', 'current_company', 'location',
    'current_start_date', 'current_years_at_company',
    'all_titles', 'all_employers', 'all_schools', 'skills', 'enriched_at'
  ];
  v_bad   text;
  v_key   text;
  v_token uuid;
  v_rows  integer;
  v_new   timestamptz;
begin
  -- Input checks --------------------------------------------------------------
  if p_linkedin_url is null or btrim(p_linkedin_url) = '' then
    raise exception 'supanova_replace_profile: p_linkedin_url is required';
  end if;
  if p_expected_updated_at is null then
    raise exception 'supanova_replace_profile: p_expected_updated_at is required';
  end if;
  if p_cols is null or jsonb_typeof(p_cols) <> 'object' or p_cols = '{}'::jsonb then
    raise exception 'supanova_replace_profile: p_cols must be a non-empty json object';
  end if;

  select string_agg(k, ', ' order by k) into v_bad
    from jsonb_object_keys(p_cols) as k
   where k <> all (c_allowed);
  if v_bad is not null then
    raise exception 'supanova_replace_profile: key(s) not allowed in p_cols: %', v_bad;
  end if;

  if p_cols ? 'raw_data' and jsonb_typeof(p_cols -> 'raw_data') <> 'object' then
    raise exception 'supanova_replace_profile: raw_data must be a json object';
  end if;

  foreach v_key in array array['name', 'current_title', 'current_company', 'location',
                               'current_start_date', 'enriched_at'] loop
    if p_cols ? v_key and jsonb_typeof(p_cols -> v_key) not in ('string', 'null') then
      raise exception 'supanova_replace_profile: % must be a string or null', v_key;
    end if;
  end loop;

  if p_cols ? 'current_years_at_company'
     and jsonb_typeof(p_cols -> 'current_years_at_company') not in ('number', 'null') then
    raise exception 'supanova_replace_profile: current_years_at_company must be a number or null';
  end if;

  foreach v_key in array array['all_titles', 'all_employers', 'all_schools', 'skills'] loop
    if p_cols ? v_key and (
         jsonb_typeof(p_cols -> v_key) <> 'array'
         or exists (select 1 from jsonb_array_elements(p_cols -> v_key) as e
                     where jsonb_typeof(e) <> 'string')
       ) then
      raise exception 'supanova_replace_profile: % must be an array of strings', v_key;
    end if;
  end loop;

  -- Open the bypass for this one row in this one transaction -------------------
  v_token := gen_random_uuid();
  insert into profile_replace_private.replace_tokens (token, txid, linkedin_url)
  values (v_token, txid_current(), p_linkedin_url);
  perform set_config('sourcing_core.replace_token', v_token::text, true);

  update public.profiles as p
     set raw_data = case when p_cols ? 'raw_data' then p_cols -> 'raw_data' else p.raw_data end,
         name = case when p_cols ? 'name' then p_cols ->> 'name' else p.name end,
         current_title = case when p_cols ? 'current_title' then p_cols ->> 'current_title' else p.current_title end,
         current_company = case when p_cols ? 'current_company' then p_cols ->> 'current_company' else p.current_company end,
         location = case when p_cols ? 'location' then p_cols ->> 'location' else p.location end,
         current_start_date = case when p_cols ? 'current_start_date'
                                   then (p_cols ->> 'current_start_date')::timestamptz
                                   else p.current_start_date end,
         current_years_at_company = case when p_cols ? 'current_years_at_company'
                                         then (p_cols ->> 'current_years_at_company')::double precision
                                         else p.current_years_at_company end,
         enriched_at = case when p_cols ? 'enriched_at'
                            then (p_cols ->> 'enriched_at')::timestamptz
                            else p.enriched_at end,
         all_titles = case when p_cols ? 'all_titles'
                           then array(select e.v from jsonb_array_elements_text(p_cols -> 'all_titles')
                                        with ordinality as e(v, i) order by e.i)
                           else p.all_titles end,
         all_employers = case when p_cols ? 'all_employers'
                              then array(select e.v from jsonb_array_elements_text(p_cols -> 'all_employers')
                                           with ordinality as e(v, i) order by e.i)
                              else p.all_employers end,
         all_schools = case when p_cols ? 'all_schools'
                            then array(select e.v from jsonb_array_elements_text(p_cols -> 'all_schools')
                                         with ordinality as e(v, i) order by e.i)
                            else p.all_schools end,
         skills = case when p_cols ? 'skills'
                       then array(select e.v from jsonb_array_elements_text(p_cols -> 'skills')
                                    with ordinality as e(v, i) order by e.i)
                       else p.skills end
   where p.linkedin_url = p_linkedin_url
     and p.updated_at = p_expected_updated_at
  returning p.updated_at into v_new;
  get diagnostics v_rows = row_count;

  -- Close the bypass before anything else can run -------------------------------
  delete from profile_replace_private.replace_tokens where token = v_token;
  perform set_config('sourcing_core.replace_token', '', true);

  if v_rows <> 1 then
    raise exception 'supanova_replace_profile: 0 rows changed for % (row missing, or updated_at is no longer %)',
      p_linkedin_url, p_expected_updated_at
      using errcode = 'P0002';
  end if;

  return v_new;
end;
$function$;

revoke all on function public.supanova_replace_profile(text, timestamptz, jsonb) from public, anon, authenticated;
grant execute on function public.supanova_replace_profile(text, timestamptz, jsonb) to service_role;
