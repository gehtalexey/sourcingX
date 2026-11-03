-- Migration 029: public.sourcingx_upsert_profiles - the Crustdata-id dedupe
-- (027) also reads the id nested under raw_data._new_api.
--
-- STATUS: NOT YET APPLIED. Written 2026-11-03. Needs a real Codex review and
-- Alexey's go-ahead before it touches the shared Supabase database.
--
-- WHY
-- 027 reads the incoming and stored Crustdata id only from top-level
-- raw_data.person_id / raw_data.crustdata_person_id. New-API profiles store it
-- at raw_data._new_api.crustdata_person_id. Measured live 2026-11-03: 837 of
-- 3,000 recently updated rows have the id ONLY there, so those people can
-- still get a second row. Found by Codex on gehtalexey/agent-kalamata#817.
--
-- WHAT CHANGED vs 027 (everything else is identical)
-- - Incoming id: person_id, else crustdata_person_id, else
--   _new_api.crustdata_person_id (each trimmed, blank = absent).
-- - The lookup WHERE repeats EXACTLY the expression and predicate of index
--   idx_profiles_crustdata_id_any (built live 2026-11-03, valid), so it uses
--   that index instead of the 2-key idx_profiles_crustdata_id:
--     coalesce(nullif(raw_data->>'person_id',''),
--              nullif(raw_data->>'crustdata_person_id',''),
--              nullif(raw_data->'_new_api'->>'crustdata_person_id',''))
--   WHERE <same expression> IS NOT NULL
-- - Advisory locks: every Crustdata id in the batch is locked ONCE, in
--   ascending order, right after the ingestion-run INSERT and before the
--   per-element loop, outside any exception block. Two concurrent batches
--   holding the same ids in opposite order can therefore never wait on each
--   other in a cycle (Codex found that deadlock on the sibling PR
--   gehtalexey/agent-kalamata#817: the per-row lock's exception handler fell
--   back to the sent URL and inserted the duplicate). Same key prefix as
--   agent-kalamata, so both projects serialize on the same person. A lock
--   error now fails the whole call instead of silently skipping dedupe.
-- - The per-row id is read from the RAW element (v_elem), the same source the
--   lock loop reads, so the id looked up is always an id that was locked.
-- - Ordering, results and grants unchanged.
-- - Requires migration 028 (the indexes) to be applied first.
--
-- Rollback: re-apply the function body from
-- migrations/027_sourcingx_upsert_profiles_dedupe_by_crustdata_id.sql (same
-- signature, so a plain CREATE OR REPLACE restores it).

CREATE OR REPLACE FUNCTION public.sourcingx_upsert_profiles(
  p_profiles jsonb            -- array of profile objects, each with a top-level linkedin_url
                              -- and an OPTIONAL top-level alias_urls array
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
SET statement_timeout = '20s'
AS $function$
declare
  v_max_batch constant int  := 100;
  v_writer    constant text := 'sourcingx';
  v_source    constant text := 'crustdata-enrich';
  v_provider  constant text := 'crustdata';
  v_schema_default constant text := 'crustdata-2025-11-01';

  v_total    int;
  v_run      uuid := pg_catalog.gen_random_uuid();
  v_approval text := 'SOURCINGX-' || to_char(now(), 'YYYYMMDD');
  v_elem     jsonb;
  v_prof     jsonb;
  v_sent     text;
  v_pid      uuid;
  v_action   text;
  v_curl     text;
  v_err      text;
  v_aliases  text[];
  v_alias_n  int;
  v_lock_cid text;     -- one Crustdata id from the batch, for the up-front lock loop
  v_cid      text;     -- incoming Crustdata id (person_id, else crustdata_person_id, else _new_api.crustdata_person_id)
  v_target   text;     -- linkedin_url of an existing row with the same Crustdata id
  v_ident    text;     -- the identity URL actually passed to upsert_profile
  v_redir    boolean;
  v_redir_n  int := 0;
  v_results  jsonb := '[]'::jsonb;
  v_ins int := 0; v_upd int := 0; v_noop int := 0; v_rej int := 0;
  v_err_n int := 0; v_unk int := 0;
begin
  if p_profiles is null or jsonb_typeof(p_profiles) <> 'array' then
    raise exception 'sourcingx_upsert_profiles: p_profiles must be a jsonb array (got %)',
      coalesce(jsonb_typeof(p_profiles), 'null');
  end if;

  v_total := jsonb_array_length(p_profiles);
  if v_total = 0 then
    return jsonb_build_object('received', 0, 'written', 0, 'results', '[]'::jsonb);
  end if;
  if v_total > v_max_batch then
    raise exception 'sourcingx_upsert_profiles: batch of % profiles exceeds the % cap per call',
      v_total, v_max_batch;
  end if;

  insert into sourcing_raw.profile_ingestion_runs
         (id, approval_id, run_name, writer, source, provider, schema_version,
          status, input_count, started_at, notes)
  values (v_run, v_approval, v_writer || '-upsert-' || v_run::text,
          v_writer, v_source, v_provider, v_schema_default,
          'started', v_total, now(),
          'via public.sourcingx_upsert_profiles');

  -- Lock every Crustdata id the batch needs ONCE, in ascending order, before
  -- any row is saved. Fixed order means two batches can never wait on each
  -- other in a cycle (no deadlock). The key prefix is the same as
  -- agent-kalamata's, so saves from both projects serialize on one person.
  -- Deliberately OUTSIDE any exception block: a lock error fails the whole
  -- call instead of silently skipping dedupe. The id expression must match
  -- the per-row v_cid expression below exactly (both read the raw element).
  for v_lock_cid in
    select distinct btrim(coalesce(
             nullif(btrim(e.elem -> 'raw_data' ->> 'person_id'), ''),
             nullif(btrim(e.elem -> 'raw_data' ->> 'crustdata_person_id'), ''),
             nullif(btrim(e.elem -> 'raw_data' -> '_new_api' ->> 'crustdata_person_id'), ''))) as cid
      from jsonb_array_elements(p_profiles) as e(elem)
     where jsonb_typeof(e.elem) = 'object'
     order by 1
  loop
    if v_lock_cid ~ '^[0-9]+$' and v_lock_cid !~ '^0+$' then
      perform pg_advisory_xact_lock(hashtextextended('sourcing-profiles-crustdata-id:' || v_lock_cid, 0));
    end if;
  end loop;

  -- Iterate the RAW array WITH ORDINALITY so every input element gets a
  -- result row, including malformed ones.
  for v_elem in
    select e.elem
      from jsonb_array_elements(p_profiles) with ordinality as e(elem, ord)
     order by e.ord
  loop
    v_pid := null; v_action := null; v_curl := null; v_err := null; v_sent := null;
    v_aliases := null; v_alias_n := null;
    v_cid := null; v_target := null; v_ident := null; v_redir := false;

    if jsonb_typeof(v_elem) <> 'object' then
      v_action := 'rejected';
      v_err := 'local-validation: array element is not a JSON object';
    elsif coalesce(btrim(v_elem ->> 'linkedin_url'), '') = '' then
      v_action := 'rejected';
      v_err := 'local-validation: missing linkedin_url';
    else
      v_sent := v_elem ->> 'linkedin_url';
      v_prof := public.engine_strip_contact(v_elem);
      v_ident := v_sent;

      -- Same person already saved under a DIFFERENT LinkedIn URL spelling?
      -- (percent-encoded vs decoded handle, or a renamed vanity URL.) The
      -- identity trigger only compares linkedin_handle(), so without this
      -- step it inserts a second row. Match on the Crustdata id instead and,
      -- if found, save onto that existing row. The WHERE clause repeats the
      -- idx_profiles_crustdata_id_any expression and predicate EXACTLY so the
      -- planner can use the index (anything else scans the whole table).
      begin
        -- Read from the raw element (v_elem), the same source the lock loop
        -- above reads, so this id is always one that is already locked.
        v_cid := btrim(coalesce(
                   nullif(btrim(v_elem -> 'raw_data' ->> 'person_id'), ''),
                   nullif(btrim(v_elem -> 'raw_data' ->> 'crustdata_person_id'), ''),
                   nullif(btrim(v_elem -> 'raw_data' -> '_new_api' ->> 'crustdata_person_id'), '')));
        if v_cid ~ '^[0-9]+$' and v_cid !~ '^0+$' then
          -- The advisory lock for v_cid was already taken by the up-front
          -- lock loop and is held until the transaction (the whole RPC call)
          -- ends, so a concurrent save of the same person under another URL
          -- spelling waits, then finds this row. The key prefix keeps it
          -- distinct from the identity trigger's handle lock.
          select p.linkedin_url
            into v_target
            from public.profiles p
           where coalesce(nullif(p.raw_data->>'person_id',''), nullif(p.raw_data->>'crustdata_person_id',''), nullif(p.raw_data->'_new_api'->>'crustdata_person_id','')) = v_cid
             and coalesce(nullif(p.raw_data->>'person_id',''), nullif(p.raw_data->>'crustdata_person_id',''), nullif(p.raw_data->'_new_api'->>'crustdata_person_id','')) IS NOT NULL
             and sourcing_core.linkedin_handle(p.linkedin_url)
                   is distinct from sourcing_core.linkedin_handle(v_sent)
           order by (exists (select 1 from sourcing_core.profile_identity pi
                              where pi.canonical_profile_id = p.id)) desc,
                    p.created_at asc nulls last,
                    p.id asc
           limit 1;
          if nullif(btrim(v_target), '') is not null then
            v_ident := v_target;
            v_redir := true;
          end if;
        end if;
      exception when others then
        -- A lookup failure (not a lock failure: locks were taken up front,
        -- outside this block) must never break the save: fall back to v_sent.
        -- The id stays locked, so a concurrent save still serializes.
        v_err := 'crustdata-id-lookup-failed: ' || SQLERRM;
        v_ident := v_sent;
        v_redir := false;
      end;

      begin
        select t.profile_id, t.action, t.canonical_url
          into v_pid, v_action, v_curl
          from sourcing_core.upsert_profile(
                 v_run, v_ident, v_prof, v_writer, v_source, v_provider,
                 coalesce(nullif(btrim(v_prof ->> 'schema_version'), ''), v_schema_default),
                 now()) t;
      exception when others then
        v_err := coalesce(v_err || '; ', '') || SQLERRM;  -- preserved deliberately, not discarded
        v_action := 'error';
      end;

      -- Merge any secondary LinkedIn URLs for this SAME person (e.g. the
      -- obfuscated ACoAA search-result form, or a caller-supplied
      -- original_url) into original_urls, now that the identity call above
      -- has resolved which profile row this person actually is.
      -- sourcing_core.upsert_profile CANNOT do this itself: it derives the
      -- identity handle from the ONE input_linkedin_url it is given and only
      -- registers THAT url, and an ACoAA url yields a DIFFERENT handle than a
      -- vanity url — sending the obfuscated form as the identity would create
      -- a SECOND profile row for the same person instead of aliasing them.
      -- Ported from agent-kalamata migration 024, which added exactly this
      -- block after its migration 023 (the file THIS migration was modelled
      -- on) was found to silently drop aliases.
      --
      -- TRIGGER SAFETY: sourcing_core.profiles_identity_before_write gates its
      -- identity-collision branch on `tg_op = 'INSERT'`, so it never runs on
      -- UPDATE. The only UPDATE-time check is the immutability guard comparing
      -- linkedin_handle(old.linkedin_url) with linkedin_handle(new.linkedin_url).
      -- This statement sets ONLY original_urls and never assigns linkedin_url,
      -- so new.linkedin_url is byte-identical to old.linkedin_url and that
      -- exception can never fire from this path.
      if v_pid is not null and v_action in ('inserted', 'updated', 'noop')
         and (jsonb_typeof(v_elem -> 'alias_urls') = 'array' or v_redir) then
        begin
          -- Ignore non-string elements and blanks rather than failing the row.
          -- On a Crustdata-id redirect the URL the caller sent is an alias of
          -- the existing row, so it is merged in too.
          select array_agg(distinct s.a)
            into v_aliases
            from (
              select btrim(elem_val #>> '{}') as a
                from jsonb_array_elements(
                       case when jsonb_typeof(v_elem -> 'alias_urls') = 'array'
                            then v_elem -> 'alias_urls' else '[]'::jsonb end) as elem_val
               where jsonb_typeof(elem_val) = 'string'
                 and nullif(btrim(elem_val #>> '{}'), '') is not null
              union all
              select btrim(v_sent) where v_redir
            ) s;

          if v_aliases is not null and array_length(v_aliases, 1) > 0 then
            -- aliases_merged = how many aliases this payload CARRIED into the
            -- merge (post-dedup), not a before/after diff of original_urls.
            v_alias_n := array_length(v_aliases, 1);

            update public.profiles p
               set original_urls = (
                 select array_agg(distinct a)
                   from unnest(coalesce(p.original_urls, '{}'::text[]) || v_aliases) as t(a)
                  where nullif(btrim(a), '') is not null)
             where p.id = v_pid;
          else
            v_alias_n := 0;
          end if;
        exception when others then
          -- An alias-merge failure must NEVER flip a successful profile save
          -- into a failure: the save itself already succeeded, and THAT is the
          -- outcome this RPC exists to guarantee. Record the reason only.
          v_err := coalesce(v_err || '; ', '') || 'alias-merge-failed: ' || SQLERRM;
          v_alias_n := 0;
        end;
      end if;
    end if;

    if v_redir then v_redir_n := v_redir_n + 1; end if;

    case coalesce(v_action, 'rejected')
      when 'inserted' then v_ins  := v_ins  + 1;
      when 'updated'  then v_upd  := v_upd  + 1;
      when 'noop'     then v_noop := v_noop + 1;
      when 'rejected' then v_rej  := v_rej  + 1;
      when 'error'    then v_err_n := v_err_n + 1;
      else v_unk := v_unk + 1;   -- never fold an unrecognised action into rejected
    end case;

    v_results := v_results || jsonb_build_object(
      'sent_url',       v_sent,
      'canonical_url',  v_curl,
      'profile_id',     v_pid,
      'action',         coalesce(v_action, 'rejected'),
      'error',          v_err,
      'aliases_merged', v_alias_n,
      'redirected_from_crustdata_id', v_redir
    );
  end loop;

  update sourcing_raw.profile_ingestion_runs r
     set status = 'completed', finished_at = now(),
         inserted_count = v_ins, updated_count = v_upd,
         noop_count = v_noop, rejected_count = v_rej,
         conflict_count = v_err_n
   where r.id = v_run;

  return jsonb_build_object(
    'received',       v_total,
    'written',        v_ins + v_upd,
    'noop',           v_noop,
    'rejected',       v_rej,
    'errors',         v_err_n,
    'unknown_action', v_unk,
    'redirected',     v_redir_n,
    'ingestion_run',  v_run,
    'results',        v_results
  );
end
$function$;

revoke all on function public.sourcingx_upsert_profiles(jsonb) from public;
grant execute on function public.sourcingx_upsert_profiles(jsonb) to service_role;

-- Rollback: re-apply the function body from
-- migrations/027_sourcingx_upsert_profiles_dedupe_by_crustdata_id.sql.
